// FABLE RAM-dedup design cd7bdce; GPT6.1 checked current-main port.
#include "strata/core/ram_dedup.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
namespace strata::core {
namespace {
constexpr uint64_t Page=4096;
using Clock=std::chrono::steady_clock;
uint64_t us(Clock::time_point t) { return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-t).count(); }
std::string cuda_error(const char* op,cudaError_t e) { return std::string(op)+": "+cudaGetErrorString(e); }
}
bool RamDedup::requested() { const char* s=std::getenv("STRATA_RAM_DEDUP"); return s&&std::strcmp(s,"1")==0; }
uint64_t RamDedup::keep_quota_env() {
    const char* s=std::getenv("STRATA_RAM_DEDUP_KEEP_GIB"); if(!s||!*s)return ~0ull;
    const double n=std::atof(s); return n>0&&n<1024?(uint64_t)(n*1073741824.0):0;
}
uint64_t RamDedup::bounce_bytes_env() {
    const char* s=std::getenv("STRATA_RAM_DEDUP_BOUNCE_MIB");
    const long long n=s&&*s?std::atoll(s):512; return (uint64_t)std::clamp<long long>(n,16,4096)<<20;
}
bool RamDedup::fail(const std::string& reason,std::string& err) {
    if(fatal_.empty())fatal_=reason; err=fatal_; return false;
}
bool RamDedup::healthy(std::string& err) const { if(fatal_.empty())return true; err=fatal_; return false; }
std::string RamDedup::error() const { std::lock_guard<std::mutex> lk(mu_);return fatal_; }
bool RamDedup::index(int64_t l,int64_t e,size_t& i) const {
    if(l<0||e<0||l>=n_layers_||e>=n_expert_)return false; i=(size_t)(l*n_expert_+e); return true;
}
bool RamDedup::span(size_t i,uint64_t& a,uint64_t& b) const {
    return interior_span(layout_->blob_offset((int64_t)i/n_expert_,(int64_t)i%n_expert_),
                         layout_->blob_bytes((int64_t)i/n_expert_),arena_bytes_,a,b);
}
bool RamDedup::protected_span(uint64_t a,uint64_t b) const {
    for(const auto& r:registered_)if(a<r.second&&b>r.first)return true; return false;
}
bool RamDedup::select(int device,int& old,std::string& err) {
    cudaError_t e=cudaGetDevice(&old);if(e!=cudaSuccess)return fail(cuda_error("get device",e),err);
    if(device<0)device=old; if(device!=old){e=cudaSetDevice(device);if(e!=cudaSuccess)return fail(cuda_error("set device",e),err);} return true;
}
bool RamDedup::restore_device(int old,std::string& err) {
    const cudaError_t e=cudaSetDevice(old); return e==cudaSuccess?true:fail(cuda_error("restore device",e),err);
}
bool RamDedup::arm(uint8_t* base,uint64_t bytes,const std::vector<std::pair<uint64_t,uint64_t>>& ranges,
                   bool normal,int64_t layers,int64_t experts,uint64_t bounce_bytes,std::string& err) {
    std::lock_guard<std::mutex> lk(mu_);
    if(armed_)return healthy(err);
#if !defined(_WIN32)
    return fail("RAM dedup currently qualified only for Windows anonymous 4K arena",err);
#else
    if(!normal||!base||(uintptr_t)base%Page||!bytes||layers<=0||experts<=0||layers>INT64_MAX/experts)
        return fail("RAM dedup requires completed anonymous VirtualAlloc 4K arena",err);
    layout_=&strata::kernels::cpu::expert_layout();
    if(layout_->n_layers!=layers||layout_->n_expert!=experts||!layout_->max_blob)return fail("RAM dedup geometry mismatch",err);
    for(auto r:ranges)if(r.first>r.second||r.second>bytes||r.second>UINT64_MAX-(Page-1))return fail("RAM dedup invalid actual registration range",err);
    base_=base;arena_bytes_=bytes;n_layers_=layers;n_expert_=experts;registered_=ranges;
    for(auto& r:registered_){r.first=r.first/Page*Page;r.second=(r.second+Page-1)/Page*Page;}
    const size_t n=(size_t)(layers*experts);std::vector<std::atomic<uint8_t>>(n).swap(state_);
    for(auto& s:state_)s.store(Present);slot_of_.assign(n,-1);
    for(size_t i=0;i<n;++i){uint64_t a,b;if(!span(i,a,b))return fail("RAM dedup blob outside arena",err);}
    bounce_blob_=layout_->max_blob;
    const size_t slots=(size_t)std::max<uint64_t>(4,bounce_bytes/bounce_blob_);
    if(slots>INT32_MAX||slots>SIZE_MAX/bounce_blob_)return fail("RAM dedup ring overflow",err);
    bounce_bytes_=slots*bounce_blob_;
    const cudaError_t e=cudaHostAlloc((void**)&bounce_,(size_t)bounce_bytes_,cudaHostAllocPortable);
    if(e!=cudaSuccess)return fail(cuda_error("bounce alloc",e),err);
    ring_.resize(slots);armed_=true;return true;
#endif
}
bool RamDedup::present(int64_t l,int64_t e) const {
    size_t i=0;return !armed_||(index(l,e,i)&&state_[i].load(std::memory_order_acquire)==Present);
}
bool RamDedup::drop(int64_t l,int64_t e,uint64_t& released,std::string& err) {
    released=0;std::lock_guard<std::mutex> lk(mu_);if(!healthy(err))return false;
    size_t i=0;if(!armed_||!index(l,e,i))return fail("RAM dedup drop invalid expert",err);
    uint64_t a,b;if(!span(i,a,b))return fail("RAM dedup drop invalid span",err);
    if(a==b||state_[i].load()!=Present)return true;
    if(protected_span(a,b)){++kept_registered_;return true;}
#if defined(_WIN32)
    // ERROR_NOT_LOCKED is permitted: some/all pages may never have been locked.
    if(!VirtualUnlock(base_+a,(SIZE_T)(b-a))&&GetLastError()!=ERROR_NOT_LOCKED)
        return fail("RAM dedup VirtualUnlock failed: "+std::to_string(GetLastError()),err);
    if(!VirtualFree(base_+a,(SIZE_T)(b-a),MEM_DECOMMIT))
        return fail("RAM dedup decommit failed: "+std::to_string(GetLastError()),err);
#else
    return fail("RAM dedup unsupported backing",err);
#endif
    state_[i].store(Absent,std::memory_order_release);++absent_;decommitted_+=b-a;released=b-a;return true;
}
bool RamDedup::device_stream(int device,void*& stream,std::string& err) {
    if(device<0)return fail("RAM dedup invalid source device",err);
    if((size_t)device>=devices_.size())devices_.resize((size_t)device+1);
    if(!devices_[device].stream){cudaStream_t s=nullptr;const auto e=cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking);
        if(e!=cudaSuccess)return fail(cuda_error("D2H stream create",e),err);devices_[device].stream=s;}
    stream=devices_[device].stream;return true;
}
bool RamDedup::evict(int64_t l,int64_t expert,const void* source,void* caller,int device,std::string& err) {
    const auto t=Clock::now();std::lock_guard<std::mutex> lk(mu_);if(!healthy(err))return false;
    size_t i=0;if(!armed_||!index(l,expert,i))return fail("RAM dedup eviction invalid expert",err);
    const auto state=state_[i].load(std::memory_order_acquire);if(state==Present)return true;
    if(state==InFlight){
        const int32_t q=slot_of_[i];
        if(q<0||!ring_[(size_t)q].live)return fail("RAM dedup in-flight backing missing",err);
        int p=-1;if(!select(device,p,err))return false;
        const auto e=cudaStreamWaitEvent((cudaStream_t)caller,(cudaEvent_t)ring_[(size_t)q].after,0);
        const bool restored=restore_device(p,err);
        if(e!=cudaSuccess)return fail(cuda_error("D2H repeat caller wait",e),err);
        return restored;
    }
    if(!source)return fail("RAM dedup absent expert has no authoritative GPU backing",err);
    if(in_ring_==ring_.size()){++bounce_waits_;if(!land_locked(tail_,err))return false;tail_=(tail_+1)%ring_.size();--in_ring_;}
    int previous=-1;if(!select(device,previous,err))return false;if(device<0)device=previous;
    void* stream=nullptr;bool ok=device_stream(device,stream,err);Slot& s=ring_[head_];
    if(ok&&s.device!=device){
        // An old consumed slot's events are owned by its original device.
        if(s.before||s.after){int p=-1;if(!select(s.device,p,err))ok=false;
            else{cudaError_t a=s.before?cudaEventDestroy((cudaEvent_t)s.before):cudaSuccess;
                 cudaError_t b=s.after?cudaEventDestroy((cudaEvent_t)s.after):cudaSuccess;
                 if(a==cudaSuccess)s.before=nullptr;if(b==cudaSuccess)s.after=nullptr;
                 ok=restore_device(p,err)&&a==cudaSuccess&&b==cudaSuccess;
                 if(!ok)fail("RAM dedup consumed event cleanup failed",err);}}
        if(ok){cudaEvent_t a=nullptr,b=nullptr;cudaError_t e=cudaEventCreateWithFlags(&a,cudaEventDisableTiming);
            s.before=a;s.device=device;if(e==cudaSuccess)e=cudaEventCreateWithFlags(&b,cudaEventDisableTiming);s.after=b;
            if(e!=cudaSuccess)ok=fail(cuda_error("D2H event create",e),err);}
    }
    cudaError_t e=cudaSuccess;
    if(ok){cudaStream_t ds=(cudaStream_t)stream,cs=(cudaStream_t)caller;
        e=cudaEventRecord((cudaEvent_t)s.before,cs);
        if(e==cudaSuccess)e=cudaStreamWaitEvent(ds,(cudaEvent_t)s.before,0);
        if(e==cudaSuccess)e=cudaMemcpyAsync(bounce_+head_*bounce_blob_,source,(size_t)layout_->blob_bytes(l),cudaMemcpyDeviceToHost,ds);
        if(e==cudaSuccess)e=cudaEventRecord((cudaEvent_t)s.after,ds);
        if(e==cudaSuccess)e=cudaStreamWaitEvent(cs,(cudaEvent_t)s.after,0);
        if(e!=cudaSuccess){
            // Partial issue must be quiesced; a failed fence poisons and retains the ring.
            const auto q=cudaStreamSynchronize(ds);
            ok=fail(cuda_error("D2H issue",e)+(q==cudaSuccess?"; drained":"; drain FAILED: "+std::string(cudaGetErrorString(q))),err);
        }
    }
    if(ok){s.idx=(int64_t)i;s.live=true;slot_of_[i]=(int32_t)head_;state_[i].store(InFlight,std::memory_order_release);
        head_=(head_+1)%ring_.size();++in_ring_;++d2h_;d2h_us_+=us(t);}
    return restore_device(previous,err)&&ok;
}
bool RamDedup::land_locked(size_t q,std::string& err) {
    Slot& s=ring_[q];if(!s.live)return true;
    const auto t=Clock::now();int p=-1;if(!select(s.device,p,err))return false;
    const auto status=cudaEventSynchronize((cudaEvent_t)s.after);const bool restored=restore_device(p,err);
    if(status!=cudaSuccess)return fail(cuda_error("D2H completion",status),err);if(!restored)return false;
    if(s.idx<0||slot_of_[(size_t)s.idx]!=(int32_t)q||state_[(size_t)s.idx].load()!=InFlight)
        return fail("RAM dedup ring ownership mismatch",err);
    uint64_t a,b;if(!span((size_t)s.idx,a,b)||protected_span(a,b))return fail("RAM dedup landing invalid span",err);
#if defined(_WIN32)
    if(VirtualAlloc(base_+a,(SIZE_T)(b-a),MEM_COMMIT,PAGE_READWRITE)!=base_+a)
        return fail("RAM dedup recommit failed: "+std::to_string(GetLastError()),err);
#endif
    const uint64_t off=layout_->blob_offset(s.idx/n_expert_,s.idx%n_expert_);
    // Shared boundary pages stay untouched: no write/write or write/read race with neighbours.
    std::memcpy(base_+a,bounce_+q*bounce_blob_+(a-off),(size_t)(b-a));
#if defined(_WIN32)
    (void)VirtualLock(base_+a,(SIZE_T)(b-a)); // performance only; committed contents remain authoritative on failure
#endif
    state_[(size_t)s.idx].store(Present,std::memory_order_release);slot_of_[(size_t)s.idx]=-1;
    s.live=false;s.idx=-1;--absent_;decommitted_-=b-a;d2h_us_+=us(t);return true;
}
bool RamDedup::ensure(int64_t l,int64_t e,std::string& err) {
    std::lock_guard<std::mutex> lk(mu_);if(!healthy(err))return false;size_t i=0;
    if(!armed_||!index(l,e,i))return fail("RAM dedup ensure invalid expert",err);
    if(state_[i].load()==Present)return true;
    if(state_[i].load()==InFlight&&slot_of_[i]>=0)return land_locked((size_t)slot_of_[i],err);
    return fail("RAM dedup expert absent without D2H recovery; disk fallback forbidden",err);
}
bool RamDedup::drain_locked(bool wait,std::string& err) {
    while(in_ring_){Slot& s=ring_[tail_];if(s.live&&!wait){int p=-1;if(!select(s.device,p,err))return false;
        const auto e=cudaEventQuery((cudaEvent_t)s.after);if(!restore_device(p,err))return false;
        if(e==cudaErrorNotReady)break;if(e!=cudaSuccess)return fail(cuda_error("D2H poll",e),err);}
        if(!land_locked(tail_,err))return false;tail_=(tail_+1)%ring_.size();--in_ring_;}return true;
}
bool RamDedup::drain(bool wait,std::string& err){std::lock_guard<std::mutex> lk(mu_);return healthy(err)&&drain_locked(wait,err);}
bool RamDedup::sync_locked(int device,std::string& err){
    for(size_t d=0;d<devices_.size();++d){if(!devices_[d].stream||(device>=0&&(int)d!=device))continue;
        int p=-1;if(!select((int)d,p,err))return false;const auto e=cudaStreamSynchronize((cudaStream_t)devices_[d].stream);
        if(!restore_device(p,err))return false;if(e!=cudaSuccess)return fail(cuda_error("D2H sync",e),err);}return true;
}
bool RamDedup::sync(int device,std::string& err){std::lock_guard<std::mutex> lk(mu_);return healthy(err)&&sync_locked(device,err);}
bool RamDedup::close(std::string& err){
    std::lock_guard<std::mutex> lk(mu_);if(!bounce_)return true;
    // Even poisoned objects must prove quiescence before freeing buffers/events.
    if(!sync_locked(-1,err))return false;
    for(auto& s:ring_){int p=-1;if(s.device<0)continue;if(!select(s.device,p,err))return false;
        const auto a=s.before?cudaEventDestroy((cudaEvent_t)s.before):cudaSuccess;
        const auto b=s.after?cudaEventDestroy((cudaEvent_t)s.after):cudaSuccess;
        if(a==cudaSuccess)s.before=nullptr;if(b==cudaSuccess)s.after=nullptr;
        const bool ok=restore_device(p,err);if(a!=cudaSuccess||b!=cudaSuccess||!ok)return fail("RAM dedup close event failure; handles retained",err);}
    for(size_t d=0;d<devices_.size();++d){if(!devices_[d].stream)continue;int p=-1;if(!select((int)d,p,err))return false;
        const auto e=cudaStreamDestroy((cudaStream_t)devices_[d].stream);if(e==cudaSuccess)devices_[d].stream=nullptr;
        if(!restore_device(p,err))return false;if(e!=cudaSuccess)return fail(cuda_error("D2H stream destroy",e),err);}
    const auto e=cudaFreeHost(bounce_);if(e!=cudaSuccess)return fail(cuda_error("D2H bounce free",e),err);
    bounce_=nullptr;armed_=false;return true;
}
RamDedup::~RamDedup(){std::string err;if(!close(err))std::fprintf(stderr,"strata ram dedup: fatal cleanup retained resources: %s\n",err.c_str());}
} // namespace strata::core
