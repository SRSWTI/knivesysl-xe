// Real-kernel scalar/XMX comparison and fragmented flat/paged equality gate.
// Build against the candidate shared library with icpx -fsycl -O2 -Ixpu/src.
// Arguments: prefix positions, query rows, page size, timed repeats.
#include "tq_common.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

struct Buffers {
    std::vector<void *> ptrs;
    template<class T> T *upload(const std::vector<T> &v) {
        auto *p = static_cast<T *>(tq_dev_alloc(v.size()*sizeof(T), "attention smoke"));
        ptrs.push_back(p); tq_h2d(p, v.data(), v.size()*sizeof(T)); return p;
    }
    ~Buffers() { tq_q().wait_and_throw(); for (auto *p:ptrs) tq_dev_free(p); }
};
static uint16_t bf16(float x) { uint32_t b; std::memcpy(&b,&x,4); b += 0x7fff + ((b>>16)&1); return uint16_t(b>>16); }
static uint16_t fp16(float x) { return sycl::bit_cast<uint16_t>(sycl::half(x)); }
int main(int argc, char **argv) {
    const int pos = argc>1 ? std::atoi(argv[1]) : 121;
    const int count = argc>2 ? std::atoi(argv[2]) : 24;
    const int page = argc>3 ? std::atoi(argv[3]) : 128;
    const int repeats = argc>4 ? std::atoi(argv[4]) : 3;
    if (pos<0 || pos>262144 || count<1 || count>4096 ||
        (page!=128 && page!=256) || repeats<1 || repeats>100) return 2;
    const int nh=24,nkv=4,hd=256,total=pos+count,blocks=(total+page-1)/page,padded=blocks*page;
    Buffers buffers; std::mt19937 rng(391);
    std::uniform_real_distribution<float> random(-1.0f,1.0f);
    std::vector<float> q(size_t(count)*nh*2*hd), zeros(size_t(count)*nh*hd);
    std::vector<uint16_t> norm(hd), ks(size_t(padded)*nkv),vs(ks.size());
    std::vector<uint8_t> k(size_t(padded)*nkv*hd),v(k.size());
    for(auto &x:q) x=random(rng);
    for(auto &x:norm) x=bf16(random(rng)*0.25f);
    for(size_t i=0;i<ks.size();++i) { ks[i]=fp16(i%31==0?0.0f:0.015f+0.035f*float(i%23)/22); vs[i]=fp16(i%29==0?0.0f:0.01f+0.08f*float(i%17)/16); }
    for(size_t i=0;i<k.size();++i) { k[i]=uint8_t((rng()%96)|((rng()&1)<<7)); v[i]=uint8_t((rng()%96)|((rng()&1)<<7)); }
    auto *dq=buffers.upload(q); auto *dn=buffers.upload(norm); auto *dk=buffers.upload(k); auto *dv=buffers.upload(v);
    auto *dks=buffers.upload(ks); auto *dvs=buffers.upload(vs); auto *out=buffers.upload(zeros);
    auto run = [&](const char *selector, tq_kv_layout_t layout) {
        setenv("TQ_XPU_PREFILL_XMX", selector,1);
        auto launch=[&] { x_prefill_attn(out,dq,dn,dk,dv,dks,dvs,pos,count,nh,nkv,hd,1e-6f,10000000.0f,0.25f,layout); tq_q().wait_and_throw(); };
        launch();
        auto start=std::chrono::steady_clock::now(); for(int r=0;r<repeats;++r) launch();
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeats;
        std::vector<float> result(zeros.size()); tq_d2h(result.data(),out,result.size()*4); return std::make_pair(ms,result);
    };
    auto reference=run("0",{nullptr,0,0}); auto matrix=run("1",{nullptr,0,0});
    unsigned long long auto_before[8]{}, auto_after[8]{};
    if(qwn_attn_branch_counts(auto_before,8)) return 4;
    auto automatic=run("auto",{nullptr,0,0});
    if(qwn_attn_branch_counts(auto_after,8)) return 4;
    const bool auto_xmx=auto_after[6]>auto_before[6];
    const bool auto_scalar=auto_after[7]>auto_before[7];
    const bool auto_exact=(auto_xmx!=auto_scalar) &&
        std::memcmp(automatic.second.data(),(auto_xmx?matrix.second:reference.second).data(),zeros.size()*4)==0;
    double aa=0,bb=0,ab=0,dd=0,maxerr=0,maxref=0;
    for(size_t i=0;i<zeros.size();++i) { double a=reference.second[i],b=matrix.second[i],d=a-b; if(!std::isfinite(a) || !std::isfinite(b)) return 3; aa+=a*a;bb+=b*b;ab+=a*b;dd+=d*d;maxerr=std::max(maxerr,std::abs(d));maxref=std::max(maxref,std::abs(a)); }
    if (!(aa>0 && bb>0)) return 3;
    double cosine=ab/std::sqrt(aa*bb),relative=std::sqrt(dd/aa);
    std::vector<int> table(blocks); for(int b=0;b<blocks;++b) table[b]=blocks-1-b;
    std::vector<uint8_t> pk(k.size()),pv(v.size()); std::vector<uint16_t> pks(ks.size()),pvs(vs.size());
    for(int t=0;t<padded;++t) { size_t physical=size_t(table[t/page])*page+t%page; std::copy_n(k.data()+size_t(t)*nkv*hd,nkv*hd,pk.data()+physical*nkv*hd); std::copy_n(v.data()+size_t(t)*nkv*hd,nkv*hd,pv.data()+physical*nkv*hd); std::copy_n(ks.data()+size_t(t)*nkv,nkv,pks.data()+physical*nkv); std::copy_n(vs.data()+size_t(t)*nkv,nkv,pvs.data()+physical*nkv); }
    tq_h2d(dk,pk.data(),pk.size()); tq_h2d(dv,pv.data(),pv.size()); tq_h2d(dks,pks.data(),pks.size()*2); tq_h2d(dvs,pvs.data(),pvs.size()*2);
    auto *dt=buffers.upload(table); auto paged=run("1",{dt,page==128?7:8,page-1});
    bool exact=std::memcmp(matrix.second.data(),paged.second.data(),zeros.size()*4)==0;
    std::vector<float> q2=q;
    for(float &value:q2) value=-value+0.125f;
    auto *dq2=buffers.upload(q2); auto *out2=buffers.upload(zeros);
    auto *dk2=buffers.upload(pk); auto *dv2=buffers.upload(pv);
    auto *dks2=buffers.upload(pks); auto *dvs2=buffers.upload(pvs);
    auto *dq1=dq; auto *out1=out; auto *dk1=dk; auto *dv1=dv;
    auto *dks1=dks; auto *dvs1=dvs;
    dq=dq2;out=out2;dk=dk2;dv=dv2;dks=dks2;dvs=dvs2;
    const tq_kv_layout_t paged_layout{dt,page==128?7:8,page-1};
    auto second_scalar=run("0",paged_layout);
    auto second_matrix=run("1",paged_layout);
    dq=dq1;out=out1;dk=dk1;dv=dv1;dks=dks1;dvs=dvs1;
    bool packed_exact=true, packed_auto_xmx=false;
    double packed_ms=0, packed_auto_ms=0;
    for(const char *selector:{"1","auto"}) {
        setenv("TQ_XPU_PREFILL_XMX",selector,1);
        unsigned long long before[8]{},after[8]{};
        if(qwn_attn_branch_counts(before,8)) return 4;
        auto launch=[&] {
            tq_prefill_attn_request_t requests[2]={
                {out1,dq1,dk1,dv1,dks1,dvs1,pos,count,paged_layout},
                {out2,dq2,dk2,dv2,dks2,dvs2,pos,count,paged_layout}};
            x_prefill_attn_packed(requests,2,dn,nh,nkv,hd,1e-6f,10000000.0f,0.25f);
            // Descriptors must be captured by value before submission returns.
            std::memset(requests,0,sizeof(requests));
            tq_q().wait_and_throw();
        };
        launch();
        auto start=std::chrono::steady_clock::now();
        for(int r=0;r<repeats;++r) launch();
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeats;
        if(qwn_attn_branch_counts(after,8)) return 4;
        const bool xmx=after[6]>before[6],scalar=after[7]>before[7];
        std::vector<float> first(zeros.size()),second(zeros.size());
        tq_d2h(first.data(),out1,first.size()*4);tq_d2h(second.data(),out2,second.size()*4);
        packed_exact &= xmx!=scalar;
        packed_exact &= std::memcmp(first.data(),(xmx?matrix.second:reference.second).data(),first.size()*4)==0;
        packed_exact &= std::memcmp(second.data(),(xmx?second_matrix.second:second_scalar.second).data(),second.size()*4)==0;
        if(selector[0]=='1') { packed_ms=ms;packed_exact &= xmx; }
        else { packed_auto_ms=ms;packed_auto_xmx=xmx; }
    }
    unsigned long long counters[8]{}; int rc=qwn_attn_branch_counts(counters,8);
    bool pass=cosine>=0.999999 && relative<=1e-4 && maxerr<=1e-4*maxref && exact && auto_exact && packed_exact && rc==0 && counters[6]>0 && counters[7]>0;
    std::cout << std::setprecision(10);
    std::cout << "{\"status\":\"" << (pass?"PASS":"FAIL") << "\",\"pos\":"<<pos<<",\"queries\":"<<count<<",\"page\":"<<page<<",\"cosine\":"<<cosine<<",\"relative_l2\":"<<relative<<",\"max_error\":"<<maxerr<<",\"max_reference\":"<<maxref<<",\"flat_paged_exact\":"<<(exact?"true":"false")<<",\"scalar_ms\":"<<reference.first<<",\"xmx_ms\":"<<matrix.first<<",\"paged_xmx_ms\":"<<paged.first<<",\"speedup\":"<<reference.first/matrix.first<<",\"xmx_calls\":"<<counters[6]<<",\"scalar_calls\":"<<counters[7]
              <<",\"auto_path\":\""<<(auto_xmx?"xmx":"scalar")<<"\",\"auto_exact\":"<<(auto_exact?"true":"false")<<",\"auto_ms\":"<<automatic.first
              <<",\"packed_exact\":"<<(packed_exact?"true":"false")<<",\"packed_ms\":"<<packed_ms<<",\"serial_pair_ms\":"<<paged.first+second_matrix.first
              <<",\"packed_auto_path\":\""<<(packed_auto_xmx?"xmx":"scalar")<<"\",\"packed_auto_ms\":"<<packed_auto_ms<<"}"<<std::endl;
    return pass?0:1;
}
