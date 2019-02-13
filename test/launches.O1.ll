; ModuleID = 'launches-host-x86_64-unknown-linux-gnu.bc'
source_filename = "launches.cu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque

$_ZN4dim3C2Ejjj = comdat any

@0 = private constant [825 x i8] c"P\EDU\BA\01\00\10\00(\03\00\00\00\00\00\00\02\00\01\01@\00\00\00h\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\07\00\01\00\14\00\00\00\00\00\00\00\00\00\00\00\15\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00P\00\00\00\00\00\00\00\00\00\00\00\C0\01\00\00\00\00\00\00\C0\00\00\00\00\00\00\00\14\05\14\00@\008\00\03\00@\00\04\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\002\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00r\00\00\00\00\00\00\002\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\02\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\C0\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\01\01H\00\00\008\00\00\00\00\00\00\005\00\00\00@\00\00\00\02\00\04\00\14\00\00\00\00\00\00\00\00\00\00\00\15 \00\00\00\00\00\00\00\00\00\00\00\00\00\003\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F0$\0A\0A\0A\0A.version 4.2\0A.target sm_20\0A.address_size 64\0A\0A\0A\0A\00\00\00\00", section ".nv_fatbin", align 8
@__cuda_fatbin_wrapper = internal constant { i32, i32, i8*, i8* } { i32 1180844977, i32 1, i8* getelementptr inbounds ([825 x i8], [825 x i8]* @0, i64 0, i64 0), i8* null }, section ".nvFatBinSegment", align 8
@__cuda_gpubin_handle = internal global i8** null, align 8
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* bitcast (void (i8*)* @__cuda_module_ctor to void ()*), i8* null }]

; Function Attrs: uwtable
define dso_local void @_Z11launches_k1v() #0 {
entry:
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
  call void @_Z2k1Pvi(i8* null, i32 4)
  br label %kcall.end

kcall.end:                                        ; preds = %kcall.configok, %entry
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp2, i32 8, i32 1, i32 1)
  %call4 = call i32 @_Z1fv()
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp3, i32 %call4, i32 1, i32 1)
  %12 = bitcast { i64, i32 }* %agg.tmp2.coerce to i8*
  %13 = bitcast %struct.dim3* %agg.tmp2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %12, i8* align 4 %13, i64 12, i1 false)
  %14 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp2.coerce, i32 0, i32 0
  %15 = load i64, i64* %14, align 4
  %16 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp2.coerce, i32 0, i32 1
  %17 = load i32, i32* %16, align 4
  %18 = bitcast { i64, i32 }* %agg.tmp3.coerce to i8*
  %19 = bitcast %struct.dim3* %agg.tmp3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %18, i8* align 4 %19, i64 12, i1 false)
  %20 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 0
  %21 = load i64, i64* %20, align 4
  %22 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 1
  %23 = load i32, i32* %22, align 4
  %call5 = call i32 @cudaConfigureCall(i64 %15, i32 %17, i64 %21, i32 %23, i64 0, %struct.CUstream_st* null)
  %tobool6 = icmp ne i32 %call5, 0
  br i1 %tobool6, label %kcall.end8, label %kcall.configok7

kcall.configok7:                                  ; preds = %kcall.end
  call void @_Z2k1Pvi(i8* null, i32 4)
  br label %kcall.end8

kcall.end8:                                       ; preds = %kcall.configok7, %kcall.end
  %call10 = call i32 @_Z1fv()
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp9, i32 %call10, i32 1, i32 1)
  %call12 = call i32 @_Z1fv()
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp11, i32 %call12, i32 1, i32 1)
  %24 = bitcast { i64, i32 }* %agg.tmp9.coerce to i8*
  %25 = bitcast %struct.dim3* %agg.tmp9 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %24, i8* align 4 %25, i64 12, i1 false)
  %26 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp9.coerce, i32 0, i32 0
  %27 = load i64, i64* %26, align 4
  %28 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp9.coerce, i32 0, i32 1
  %29 = load i32, i32* %28, align 4
  %30 = bitcast { i64, i32 }* %agg.tmp11.coerce to i8*
  %31 = bitcast %struct.dim3* %agg.tmp11 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %30, i8* align 4 %31, i64 12, i1 false)
  %32 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 0
  %33 = load i64, i64* %32, align 4
  %34 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 1
  %35 = load i32, i32* %34, align 4
  %call13 = call i32 @cudaConfigureCall(i64 %27, i32 %29, i64 %33, i32 %35, i64 0, %struct.CUstream_st* null)
  %tobool14 = icmp ne i32 %call13, 0
  br i1 %tobool14, label %kcall.end16, label %kcall.configok15

kcall.configok15:                                 ; preds = %kcall.end8
  call void @_Z2k1Pvi(i8* null, i32 4)
  br label %kcall.end16

kcall.end16:                                      ; preds = %kcall.configok15, %kcall.end8
  %call18 = call i32 @_Z1fv()
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp17, i32 %call18, i32 1, i32 1)
  %call20 = call i32 @_Z1fv()
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp19, i32 %call20, i32 1, i32 1)
  %36 = bitcast { i64, i32 }* %agg.tmp17.coerce to i8*
  %37 = bitcast %struct.dim3* %agg.tmp17 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %36, i8* align 4 %37, i64 12, i1 false)
  %38 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp17.coerce, i32 0, i32 0
  %39 = load i64, i64* %38, align 4
  %40 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp17.coerce, i32 0, i32 1
  %41 = load i32, i32* %40, align 4
  %42 = bitcast { i64, i32 }* %agg.tmp19.coerce to i8*
  %43 = bitcast %struct.dim3* %agg.tmp19 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %42, i8* align 4 %43, i64 12, i1 false)
  %44 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp19.coerce, i32 0, i32 0
  %45 = load i64, i64* %44, align 4
  %46 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp19.coerce, i32 0, i32 1
  %47 = load i32, i32* %46, align 4
  %call21 = call i32 @cudaConfigureCall(i64 %39, i32 %41, i64 %45, i32 %47, i64 0, %struct.CUstream_st* null)
  %tobool22 = icmp ne i32 %call21, 0
  br i1 %tobool22, label %kcall.end26, label %kcall.configok23

kcall.configok23:                                 ; preds = %kcall.end16
  %call24 = call i8* @_Z1gv()
  %call25 = call i32 @_Z1fv()
  call void @_Z2k1Pvi(i8* %call24, i32 %call25)
  br label %kcall.end26

kcall.end26:                                      ; preds = %kcall.configok23, %kcall.end16
  %call28 = call i32 @_Z1fv()
  %call29 = call i32 @_Z1fv()
  %add = add nsw i32 %call28, %call29
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp27, i32 %add, i32 1, i32 1)
  %call31 = call i32 @_Z1fv()
  %mul = mul nsw i32 %call31, 2
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp30, i32 %mul, i32 1, i32 1)
  %48 = bitcast { i64, i32 }* %agg.tmp27.coerce to i8*
  %49 = bitcast %struct.dim3* %agg.tmp27 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %48, i8* align 4 %49, i64 12, i1 false)
  %50 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp27.coerce, i32 0, i32 0
  %51 = load i64, i64* %50, align 4
  %52 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp27.coerce, i32 0, i32 1
  %53 = load i32, i32* %52, align 4
  %54 = bitcast { i64, i32 }* %agg.tmp30.coerce to i8*
  %55 = bitcast %struct.dim3* %agg.tmp30 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %54, i8* align 4 %55, i64 12, i1 false)
  %56 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp30.coerce, i32 0, i32 0
  %57 = load i64, i64* %56, align 4
  %58 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp30.coerce, i32 0, i32 1
  %59 = load i32, i32* %58, align 4
  %call32 = call i32 @cudaConfigureCall(i64 %51, i32 %53, i64 %57, i32 %59, i64 0, %struct.CUstream_st* null)
  %tobool33 = icmp ne i32 %call32, 0
  br i1 %tobool33, label %kcall.end39, label %kcall.configok34

kcall.configok34:                                 ; preds = %kcall.end26
  %call35 = call i8* @_Z1gv()
  %call36 = call i32 @_Z1fv()
  %call37 = call i32 @_Z1fv()
  %add38 = add nsw i32 %call36, %call37
  call void @_Z2k1Pvi(i8* %call35, i32 %add38)
  br label %kcall.end39

kcall.end39:                                      ; preds = %kcall.configok34, %kcall.end26
  ret void
}

declare dso_local i32 @cudaConfigureCall(i64, i32, i64, i32, i64, %struct.CUstream_st*) #1

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3* %this, i32 %vx, i32 %vy, i32 %vz) unnamed_addr #2 comdat align 2 {
entry:
  %this.addr = alloca %struct.dim3*, align 8
  %vx.addr = alloca i32, align 4
  %vy.addr = alloca i32, align 4
  %vz.addr = alloca i32, align 4
  store %struct.dim3* %this, %struct.dim3** %this.addr, align 8, !tbaa !2
  store i32 %vx, i32* %vx.addr, align 4, !tbaa !6
  store i32 %vy, i32* %vy.addr, align 4, !tbaa !6
  store i32 %vz, i32* %vz.addr, align 4, !tbaa !6
  %this1 = load %struct.dim3*, %struct.dim3** %this.addr, align 8
  %x = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 0
  %0 = load i32, i32* %vx.addr, align 4, !tbaa !6
  store i32 %0, i32* %x, align 4, !tbaa !8
  %y = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 1
  %1 = load i32, i32* %vy.addr, align 4, !tbaa !6
  store i32 %1, i32* %y, align 4, !tbaa !10
  %z = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 2
  %2 = load i32, i32* %vz.addr, align 4, !tbaa !6
  store i32 %2, i32* %z, align 4, !tbaa !11
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #3

declare dso_local void @_Z2k1Pvi(i8*, i32) #1

declare dso_local i32 @_Z1fv() #1

declare dso_local i8* @_Z1gv() #1

declare dso_local i8** @__cudaRegisterFatBinary(i8*)

define internal void @__cuda_module_ctor(i8*) {
entry:
  %1 = call i8** @__cudaRegisterFatBinary(i8* bitcast ({ i32, i32, i8*, i8* }* @__cuda_fatbin_wrapper to i8*))
  store i8** %1, i8*** @__cuda_gpubin_handle, align 8
  %2 = call i32 @atexit(void (i8*)* @__cuda_module_dtor)
  ret void
}

declare dso_local void @__cudaUnregisterFatBinary(i8**)

define internal void @__cuda_module_dtor(i8*) {
entry:
  %1 = load i8**, i8*** @__cuda_gpubin_handle, align 8
  call void @__cudaUnregisterFatBinary(i8** %1)
  ret void
}

declare dso_local i32 @atexit(void (i8*)*)

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.0.0 (https://github.com/llvm-mirror/clang f3e3e0682c76f3e027ce0e21c05049ff48ccdd6f) (https://github.com/llvm-mirror/llvm a5b9a59eac7814d1276784cec74bbcfd6449d318)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !4, i64 0}
!8 = !{!9, !7, i64 0}
!9 = !{!"_ZTS4dim3", !7, i64 0, !7, i64 4, !7, i64 8}
!10 = !{!9, !7, i64 4}
!11 = !{!9, !7, i64 8}
