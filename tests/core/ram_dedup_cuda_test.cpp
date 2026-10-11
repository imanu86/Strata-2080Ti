// Real checked RAM decommit / cross-stream D2H lifecycle. No model; ~6 MiB per card.
#include "strata/core/ram_dedup.hpp"
#include "strata/core/pinned.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <chrono>
#include <vector>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
int main() {
#if !defined(_WIN32) || defined(__HIPCC__)
    return 77;
#else
    // This qualification includes global landing across two real devices.
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||count<2)return 77;
    std::string e;int checks=0;
    auto ck=[&](bool x,const char* n){++checks;if(!x)std::fprintf(stderr,"FAIL %s: %s\n",n,e.c_str());return x;};
    // Actual current allocator, not a mirror of its mapping bookkeeping.
    _putenv_s("STRATA_NO_LARGEPAGES", "1");
    {
        strata::core::PinnedArena arena(1u<<20);
        if(!ck(arena.valid() && arena.dedup_private_layout(), "actual PinnedArena anonymous admission"))return 1;
        MEMORY_BASIC_INFORMATION mbi{};
        if(!ck(VirtualQuery(arena.base,&mbi,sizeof(mbi)) && mbi.Type==MEM_PRIVATE &&
               mbi.AllocationBase==arena.base, "allocator actual private allocation"))return 1;
        const auto backing=arena.backing;
        arena.backing=strata::core::PageBacking::LargePages;
        const bool large_rejected=!arena.dedup_private_layout();
        arena.backing=strata::core::PageBacking::PinnedByCuda;
        const bool cuda_backing_rejected=!arena.dedup_private_layout();arena.backing=backing;
        auto* mapping=arena.mapping_base;arena.mapping_base=(uint8_t*)arena.base+4096;
        const bool shared_offset_rejected=!arena.dedup_private_layout();arena.mapping_base=mapping;
        const auto bytes=arena.mapping_bytes;arena.mapping_bytes=bytes-1;
        const bool truncated_rejected=!arena.dedup_private_layout();arena.mapping_bytes=bytes;
        const auto registered=arena.registered_bytes;arena.registered_bytes=arena.capacity+1;
        const bool registration_rejected=!arena.dedup_private_layout();arena.registered_bytes=registered;
        if(!ck(large_rejected && cuda_backing_rejected && shared_offset_rejected &&
               truncated_rejected && registration_rejected, "production admission negative metadata"))return 1;
        if(!ck(arena.dedup_private_layout(), "registered normal backing remains admissible"))return 1;
        strata::core::PinnedArena shared(1u<<20,{},0,"dedup-no-shared-test",73);
        if(!ck(!shared.valid() && !shared.dedup_private_layout(), "Windows real shared allocator refuses"))return 1;
    }
    namespace fs=std::filesystem;
    fs::path temp=fs::temp_directory_path()/ ("strata-dd-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(temp);
    if(!ck(strata::kernels::cpu::expert_layout_load(temp.string(),1,8,e),"canonical synthetic layout"))return 1;
    const auto& lay=strata::kernels::cpu::expert_layout();
    const size_t blob=(size_t)lay.blob_bytes(0), bytes=((blob*8+4095)/4096)*4096;
    auto* base=(unsigned char*)VirtualAlloc(nullptr,bytes,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
    if(!ck(base!=nullptr,"anonymous arena"))return 1;
    std::vector<unsigned char> original(bytes);
    for(size_t i=0;i<bytes;++i)original[i]=(unsigned char)(13+(i*19)%229);
    std::memcpy(base,original.data(),bytes);
    strata::core::RamDedup dd;
    if(!ck(dd.arm(base,bytes,{{0,4096}},true,1,8,blob*4,e),"arm real ring"))return 1;
    std::vector<void*> slots((size_t)count);std::vector<cudaStream_t> a((size_t)count),b((size_t)count);
    for(int d=0;d<count;++d){if(!ck(cudaSetDevice(d)==cudaSuccess,"select actual device"))return 1;if(!ck(cudaMalloc(&slots[(size_t)d],blob)==cudaSuccess&&
        cudaStreamCreateWithFlags(&a[(size_t)d],cudaStreamNonBlocking)==cudaSuccess&&
        cudaStreamCreateWithFlags(&b[(size_t)d],cudaStreamNonBlocking)==cudaSuccess,"device stream/backing"))return 1;}
    uint64_t freed=0;
    if(!ck(dd.drop(0,0,freed,e)&&freed==0&&dd.present(0,0),"declared protected range preserves expert"))return 1;
    for(int d=0;d<count;++d){
        const int expert=1+(d%3);const size_t off=(size_t)lay.blob_offset(0,expert);
        if(!ck(cudaSetDevice(d)==cudaSuccess,"select backing device"))return 1;
        if(!ck(cudaMemcpyAsync(slots[(size_t)d],base+off,blob,cudaMemcpyHostToDevice,a[(size_t)d])==cudaSuccess&&
               cudaStreamSynchronize(a[(size_t)d])==cudaSuccess,"fill before drop"))return 1;
        if(!ck(dd.drop(0,expert,freed,e)&&freed>0&&!dd.present(0,expert),"real decommit"))return 1;
        MEMORY_BASIC_INFORMATION mbi{};uint64_t lo=0,hi=0;strata::core::RamDedup::interior_span(off,blob,bytes,lo,hi);
        VirtualQuery(base+lo,&mbi,sizeof(mbi));if(!ck(mbi.State==MEM_RESERVE,"OS confirms decommitted page"))return 1;
        if(!ck(dd.evict(0,expert,slots[(size_t)d],a[(size_t)d],d,e)&&
               dd.evict(0,expert,slots[(size_t)d],b[(size_t)d],d,e),"two callers wait on same D2H"))return 1;
        // Both overwrites must be ordered after the recovery copy. No external
        // synchronize is inserted between evict and these destructive writes.
        if(!ck(cudaMemsetAsync(slots[(size_t)d],0xE7,blob,b[(size_t)d])==cudaSuccess&&
               cudaMemsetAsync(slots[(size_t)d],0x39,blob,a[(size_t)d])==cudaSuccess&&
               cudaStreamSynchronize(a[(size_t)d])==cudaSuccess&&cudaStreamSynchronize(b[(size_t)d])==cudaSuccess,
               "immediate overwrite on both callers"))return 1;
        if(!ck(dd.ensure(0,expert,e)&&std::memcmp(base,original.data(),bytes)==0,
               "recovered bytes and all neighbour edges identical"))return 1;
    }
    // V4 publication order: sync-only leaves RAM unavailable; global landing
    // makes both victims CPU-readable BEFORE their GPU backing is retired.
    // Use real Windows pages and production RamDedup, never ensure here.
    auto pages_are=[&](int expert,DWORD state) {
        const uint64_t off=lay.blob_offset(0,expert);
        uint64_t lo=0,hi=0;
        if(!strata::core::RamDedup::interior_span(off,blob,bytes,lo,hi)||lo==hi)return false;
        for(uint64_t q=lo;q<hi;q+=4096) {
            MEMORY_BASIC_INFORMATION mbi{};
            if(VirtualQuery(base+q,&mbi,sizeof(mbi))!=sizeof(mbi)||mbi.State!=state||
               mbi.Type!=MEM_PRIVATE||mbi.AllocationBase!=base)return false;
        }
        return true;
    };
    const int victim[2]={4,5};
    const auto recoveries_before=dd.d2h_experts();
    for(int d=0;d<2;++d) {
        const size_t off=(size_t)lay.blob_offset(0,victim[d]);
        if(!ck(cudaSetDevice(d)==cudaSuccess&&
               cudaMemcpyAsync(slots[(size_t)d],base+off,blob,cudaMemcpyHostToDevice,a[(size_t)d])==cudaSuccess&&
               cudaStreamSynchronize(a[(size_t)d])==cudaSuccess,"publish-order authoritative GPU fill"))return 1;
        if(!ck(dd.drop(0,victim[d],freed,e)&&freed>0&&pages_are(victim[d],MEM_RESERVE),
               "publish-order actual interior decommit"))return 1;
        if(!ck(dd.evict(0,victim[d],slots[(size_t)d],a[(size_t)d],d,e),
               "publish-order queue both device recoveries before landing"))return 1;
    }
    for(int d=0;d<2;++d) {
        if(!ck(dd.sync(d,e)&&!dd.present(0,victim[d])&&pages_are(victim[d],MEM_RESERVE),
               "completed D2H alone does not publish Present RAM"))return 1;
    }
    if(!ck(dd.drain(true,e),"global landing before residency retirement"))return 1;
    for(int d=0;d<2;++d) {
        if(!ck(dd.present(0,victim[d])&&pages_are(victim[d],MEM_COMMIT),
               "global landing commits every interior page on both devices"))return 1;
    }
    if(!ck(std::memcmp(base,original.data(),bytes)==0,
           "CPU reads exact victims/protected pages/boundary bytes without ensure"))return 1;
    // Destructive overwrite represents reusing the retired GPU slot. No RAM
    // recovery is allowed after this point: both host copies must already exist.
    for(int d=0;d<2;++d) {
        if(!ck(cudaSetDevice(d)==cudaSuccess&&
               cudaMemsetAsync(slots[(size_t)d],0xA4+d,blob,a[(size_t)d])==cudaSuccess&&
               cudaStreamSynchronize(a[(size_t)d])==cudaSuccess,
               "overwrite retired GPU backing only after Present landing"))return 1;
    }
    if(!ck(dd.present(0,4)&&dd.present(0,5)&&dd.present(0,0)&&
           std::memcmp(base,original.data(),bytes)==0&&dd.d2h_experts()==recoveries_before+2,
           "CPU RAM survives GPU retirement with no ensure or extra D2H"))return 1;
    // Force ring wrap/backpressure and event re-ownership across devices.
    // 8 distinct experts permit more than four live recovery entries.
    for(int phase=0;phase<2;++phase) {
        for(int k=0;k<8;++k) {
            const int d=(k+phase)%count, expert=1+(k%7);
            const size_t off=(size_t)lay.blob_offset(0,expert);
            if(!ck(cudaSetDevice(d)==cudaSuccess,"ring select device"))return 1;
            if(!ck(cudaMemcpyAsync(slots[(size_t)d],base+off,blob,cudaMemcpyHostToDevice,a[(size_t)d])==cudaSuccess&&
                   cudaStreamSynchronize(a[(size_t)d])==cudaSuccess,"ring backing fill"))return 1;
            if(!ck(dd.drop(0,expert,freed,e)&&dd.evict(0,expert,slots[(size_t)d],a[(size_t)d],d,e),
                   "ring recovery queue without immediate ensure"))return 1;
        }
        if(!ck(dd.drain(true,e)&&std::memcmp(base,original.data(),bytes)==0,"wrapped ring byte verification"))return 1;
    }
    if(!ck(dd.bounce_waits()>0,"actual ring backpressure exercised"))return 1;
    if(!ck(dd.drain(true,e)&&dd.close(e),"checked drain/close"))return 1;
    // A real invalid device API must poison admission before overwrite; no SSD fallback.
    strata::core::RamDedup bad;
    if(!ck(bad.arm(base,bytes,{},true,1,8,blob*4,e),"fault test arm"))return 1;
    if(!ck(bad.drop(0,1,freed,e)&&!bad.evict(0,1,slots[0],a[0],count+20,e)&&
           !bad.drop(0,2,freed,e)&&!bad.ensure(0,1,e),"invalid device is sticky fatal"))return 1;
    cudaGetLastError();if(!ck(bad.close(e),"fault cleanup checked"))return 1;
    for(int d=0;d<count;++d){if(!ck(cudaSetDevice(d)==cudaSuccess&&cudaStreamDestroy(a[(size_t)d])==cudaSuccess&&cudaStreamDestroy(b[(size_t)d])==cudaSuccess&&cudaFree(slots[(size_t)d])==cudaSuccess,"checked device cleanup"))return 1;}
    VirtualFree(base,0,MEM_RELEASE);std::error_code ignored;fs::remove_all(temp,ignored);
    std::printf("PASS RAM dedup actual CUDA/Windows %d checks, %d devices\n",checks,count);return 0;
#endif
}
