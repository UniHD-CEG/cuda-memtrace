extern int f();
extern float* g();
extern void h();

__global__
void k1(float *a, int b) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  a[gid] += b;
}

void launches_k1(float *a) {
  k1<<<8, 8>>>(a, 4);
  k1<<<8, f()>>>(a, 4);
  k1<<<f(), f()>>>(a, 4);
  k1<<<f(), f()>>>(g(), f());
  k1<<<f()+f(), f()*2>>>(g(), f()+f());
}
