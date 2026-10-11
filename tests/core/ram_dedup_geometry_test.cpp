// CPU test of the actual production page geometry helper, no CUDA/model.
#include "strata/core/ram_dedup.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
int main() {
    uint64_t a=0,b=0;int checks=0;
    auto ck=[&](bool x,const char* n){++checks;if(!x)std::fprintf(stderr,"FAIL %s\n",n);return x;};
    using strata::core::RamDedup;
    if(!ck(RamDedup::interior_span(123,12288,16384,a,b)&&a==4096&&b==12288,"unaligned interior"))return 1;
    std::vector<unsigned char> mem(16384,73), original=mem;
    std::memset(mem.data()+a,0,(size_t)(b-a));
    if(!ck(std::memcmp(mem.data(),original.data(),(size_t)a)==0 &&
           std::memcmp(mem.data()+b,original.data()+b,mem.size()-(size_t)b)==0,"shared neighbour edges unchanged"))return 1;
    if(!ck(RamDedup::interior_span(4096,8192,16384,a,b)&&a==4096&&b==12288,"aligned expert"))return 1;
    if(!ck(RamDedup::interior_span(1023,1024,4096,a,b)&&a==b,"subpage cannot reclaim"))return 1;
    if(!ck(!RamDedup::interior_span(16000,512,16384,a,b),"outside arena rejected"))return 1;
    if(!ck(!RamDedup::interior_span(UINT64_MAX-100,200,UINT64_MAX,a,b),"overflow rejected"))return 1;
    if(!ck(!RamDedup::interior_span(UINT64_MAX-1,0,UINT64_MAX,a,b),"ceil overflow rejected"))return 1;
    if(!ck(RamDedup::interior_span(0,0,4096,a,b)&&a==0&&b==0,"empty span"))return 1;
    std::printf("PASS actual geometry %d checks\n",checks);return 0;
}
