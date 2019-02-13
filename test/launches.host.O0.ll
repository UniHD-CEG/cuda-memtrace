; ModuleID = 'launches.cu'
source_filename = "launches.cu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque

$_ZN4dim3C2Ejjj = comdat any

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z2k1Pfi(float* %a, i32 %b) #0 {
entry:
  %a.addr = alloca float*, align 8
  %b.addr = alloca i32, align 4
  store float* %a, float** %a.addr, align 8
  store i32 %b, i32* %b.addr, align 4
  %0 = bitcast float** %a.addr to i8*
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0)
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %setup.next, label %setup.end

setup.next:                                       ; preds = %entry
  %3 = bitcast i32* %b.addr to i8*
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 4, i64 8)
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %setup.next1, label %setup.end

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (float*, i32)* @_Z2k1Pfi to i8*))
  br label %setup.end

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void
}

declare dso_local i32 @cudaSetupArgument(i8*, i64, i64)

declare dso_local i32 @cudaLaunch(i8*)

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z11launches_k1Pf(float* %a) #0 {
entry:
  %a.addr = alloca float*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %agg.tmp2 = alloca %struct.dim3, align 4
  %agg.tmp3 = alloca %struct.dim3, align 4
  %agg.tmp2.coerce = alloca { i64, i32 }, align 4
  %agg.tmp3.coerce = alloca { i64, i32 }, align 4
  %agg.tmp9 = alloca %struct.dim3, align 4
  %agg.tmp11 = alloca %struct.dim3, align 4
  %agg.tmp9.coerce = alloca { i64, i32 }, align 4
  %agg.tmp11.coerce = alloca { i64, i32 }, align 4
  %agg.tmp17 = alloca %struct.dim3, align 4
  %agg.tmp19 = alloca %struct.dim3, align 4
  %agg.tmp17.coerce = alloca { i64, i32 }, align 4
  %agg.tmp19.coerce = alloca { i64, i32 }, align 4
  %agg.tmp27 = alloca %struct.dim3, align 4
  %agg.tmp30 = alloca %struct.dim3, align 4
  %agg.tmp27.coerce = alloca { i64, i32 }, align 4
  %agg.tmp30.coerce = alloca { i64, i32 }, align 4
  store float* %a, float** %a.addr, align 8
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 8, i32 1, i32 1)
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 8, i32 1, i32 1)
  %0 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*
  %1 = bitcast %struct.dim3* %agg.tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %0, i8* align 4 %1, i64 12, i1 false)
  %2 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0
  %3 = load i64, i64* %2, align 4
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1
  %5 = load i32, i32* %4, align 4
  %6 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*
  %7 = bitcast %struct.dim3* %agg.tmp1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %6, i8* align 4 %7, i64 12, i1 false)
  %8 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0
  %9 = load i64, i64* %8, align 4
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1
  %11 = load i32, i32* %10, align 4
  %call = call i32 @cudaConfigureCall(i64 %3, i32 %5, i64 %9, i32 %11, i64 0, %struct.CUstream_st* null)
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %kcall.end, label %kcall.configok

kcall.configok:                                   ; preds = %entry
  %12 = load float*, float** %a.addr, align 8
  call void @_Z2k1Pfi(float* %12, i32 4)
  br label %kcall.end

kcall.end:                                        ; preds = %kcall.configok, %entry
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp2, i32 8, i32 1, i32 1)
  %call4 = call i32 @_Z1fv()
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp3, i32 %call4, i32 1, i32 1)
  %13 = bitcast { i64, i32 }* %agg.tmp2.coerce to i8*
  %14 = bitcast %struct.dim3* %agg.tmp2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %13, i8* align 4 %14, i64 12, i1 false)
  %15 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp2.coerce, i32 0, i32 0
  %16 = load i64, i64* %15, align 4
  %17 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp2.coerce, i32 0, i32 1
  %18 = load i32, i32* %17, align 4
  %19 = bitcast { i64, i32 }* %agg.tmp3.coerce to i8*
  %20 = bitcast %struct.dim3* %agg.tmp3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %19, i8* align 4 %20, i64 12, i1 false)
  %21 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 0
  %22 = load i64, i64* %21, align 4
  %23 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 1
  %24 = load i32, i32* %23, align 4
  %call5 = call i32 @cudaConfigureCall(i64 %16, i32 %18, i64 %22, i32 %24, i64 0, %struct.CUstream_st* null)
  %tobool6 = icmp ne i32 %call5, 0
  br i1 %tobool6, label %kcall.end8, label %kcall.configok7

kcall.configok7:                                  ; preds = %kcall.end
  %25 = load float*, float** %a.addr, align 8
  call void @_Z2k1Pfi(float* %25, i32 4)
  br label %kcall.end8

kcall.end8:                                       ; preds = %kcall.configok7, %kcall.end
  %call10 = call i32 @_Z1fv()
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp9, i32 %call10, i32 1, i32 1)
  %call12 = call i32 @_Z1fv()
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp11, i32 %call12, i32 1, i32 1)
  %26 = bitcast { i64, i32 }* %agg.tmp9.coerce to i8*
  %27 = bitcast %struct.dim3* %agg.tmp9 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %26, i8* align 4 %27, i64 12, i1 false)
  %28 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp9.coerce, i32 0, i32 0
  %29 = load i64, i64* %28, align 4
  %30 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp9.coerce, i32 0, i32 1
  %31 = load i32, i32* %30, align 4
  %32 = bitcast { i64, i32 }* %agg.tmp11.coerce to i8*
  %33 = bitcast %struct.dim3* %agg.tmp11 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %32, i8* align 4 %33, i64 12, i1 false)
  %34 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 0
  %35 = load i64, i64* %34, align 4
  %36 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 1
  %37 = load i32, i32* %36, align 4
  %call13 = call i32 @cudaConfigureCall(i64 %29, i32 %31, i64 %35, i32 %37, i64 0, %struct.CUstream_st* null)
  %tobool14 = icmp ne i32 %call13, 0
  br i1 %tobool14, label %kcall.end16, label %kcall.configok15

kcall.configok15:                                 ; preds = %kcall.end8
  %38 = load float*, float** %a.addr, align 8
  call void @_Z2k1Pfi(float* %38, i32 4)
  br label %kcall.end16

kcall.end16:                                      ; preds = %kcall.configok15, %kcall.end8
  %call18 = call i32 @_Z1fv()
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp17, i32 %call18, i32 1, i32 1)
  %call20 = call i32 @_Z1fv()
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp19, i32 %call20, i32 1, i32 1)
  %39 = bitcast { i64, i32 }* %agg.tmp17.coerce to i8*
  %40 = bitcast %struct.dim3* %agg.tmp17 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %39, i8* align 4 %40, i64 12, i1 false)
  %41 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp17.coerce, i32 0, i32 0
  %42 = load i64, i64* %41, align 4
  %43 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp17.coerce, i32 0, i32 1
  %44 = load i32, i32* %43, align 4
  %45 = bitcast { i64, i32 }* %agg.tmp19.coerce to i8*
  %46 = bitcast %struct.dim3* %agg.tmp19 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %45, i8* align 4 %46, i64 12, i1 false)
  %47 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp19.coerce, i32 0, i32 0
  %48 = load i64, i64* %47, align 4
  %49 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp19.coerce, i32 0, i32 1
  %50 = load i32, i32* %49, align 4
  %call21 = call i32 @cudaConfigureCall(i64 %42, i32 %44, i64 %48, i32 %50, i64 0, %struct.CUstream_st* null)
  %tobool22 = icmp ne i32 %call21, 0
  br i1 %tobool22, label %kcall.end26, label %kcall.configok23

kcall.configok23:                                 ; preds = %kcall.end16
  %call24 = call float* @_Z1gv()
  %call25 = call i32 @_Z1fv()
  call void @_Z2k1Pfi(float* %call24, i32 %call25)
  br label %kcall.end26

kcall.end26:                                      ; preds = %kcall.configok23, %kcall.end16
  %call28 = call i32 @_Z1fv()
  %call29 = call i32 @_Z1fv()
  %add = add nsw i32 %call28, %call29
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp27, i32 %add, i32 1, i32 1)
  %call31 = call i32 @_Z1fv()
  %mul = mul nsw i32 %call31, 2
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp30, i32 %mul, i32 1, i32 1)
  %51 = bitcast { i64, i32 }* %agg.tmp27.coerce to i8*
  %52 = bitcast %struct.dim3* %agg.tmp27 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %51, i8* align 4 %52, i64 12, i1 false)
  %53 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp27.coerce, i32 0, i32 0
  %54 = load i64, i64* %53, align 4
  %55 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp27.coerce, i32 0, i32 1
  %56 = load i32, i32* %55, align 4
  %57 = bitcast { i64, i32 }* %agg.tmp30.coerce to i8*
  %58 = bitcast %struct.dim3* %agg.tmp30 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %57, i8* align 4 %58, i64 12, i1 false)
  %59 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp30.coerce, i32 0, i32 0
  %60 = load i64, i64* %59, align 4
  %61 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp30.coerce, i32 0, i32 1
  %62 = load i32, i32* %61, align 4
  %call32 = call i32 @cudaConfigureCall(i64 %54, i32 %56, i64 %60, i32 %62, i64 0, %struct.CUstream_st* null)
  %tobool33 = icmp ne i32 %call32, 0
  br i1 %tobool33, label %kcall.end39, label %kcall.configok34

kcall.configok34:                                 ; preds = %kcall.end26
  %call35 = call float* @_Z1gv()
  %call36 = call i32 @_Z1fv()
  %call37 = call i32 @_Z1fv()
  %add38 = add nsw i32 %call36, %call37
  call void @_Z2k1Pfi(float* %call35, i32 %add38)
  br label %kcall.end39

kcall.end39:                                      ; preds = %kcall.configok34, %kcall.end26
  ret void
}

declare dso_local i32 @cudaConfigureCall(i64, i32, i64, i32, i64, %struct.CUstream_st*) #1

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3* %this, i32 %vx, i32 %vy, i32 %vz) unnamed_addr #2 comdat align 2 {
entry:
  %this.addr = alloca %struct.dim3*, align 8
  %vx.addr = alloca i32, align 4
  %vy.addr = alloca i32, align 4
  %vz.addr = alloca i32, align 4
  store %struct.dim3* %this, %struct.dim3** %this.addr, align 8
  store i32 %vx, i32* %vx.addr, align 4
  store i32 %vy, i32* %vy.addr, align 4
  store i32 %vz, i32* %vz.addr, align 4
  %this1 = load %struct.dim3*, %struct.dim3** %this.addr, align 8
  %x = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 0
  %0 = load i32, i32* %vx.addr, align 4
  store i32 %0, i32* %x, align 4
  %y = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 1
  %1 = load i32, i32* %vy.addr, align 4
  store i32 %1, i32* %y, align 4
  %z = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 2
  %2 = load i32, i32* %vz.addr, align 4
  store i32 %2, i32* %z, align 4
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #3

declare dso_local i32 @_Z1fv() #1

declare dso_local float* @_Z1gv() #1

attributes #0 = { noinline optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.0.0 (https://github.com/llvm-mirror/clang f3e3e0682c76f3e027ce0e21c05049ff48ccdd6f) (https://github.com/llvm-mirror/llvm a5b9a59eac7814d1276784cec74bbcfd6449d318)"}
