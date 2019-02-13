; ModuleID = 'launches.cu'
source_filename = "launches.cu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.CUstream_st = type opaque

; Function Attrs: uwtable
define dso_local void @_Z2k1Pfi(float* %a, i32 %b) #0 {
entry:
  %a.addr = alloca float*, align 8
  %b.addr = alloca i32, align 4
  store float* %a, float** %a.addr, align 8, !tbaa !2
  store i32 %b, i32* %b.addr, align 4, !tbaa !6
  %0 = bitcast float** %a.addr to i8*
  %1 = call i32 @cudaSetupArgument(i8* nonnull %0, i64 8, i64 0)
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %setup.next, label %setup.end

setup.next:                                       ; preds = %entry
  %3 = bitcast i32* %b.addr to i8*
  %4 = call i32 @cudaSetupArgument(i8* nonnull %3, i64 4, i64 8)
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %setup.next1, label %setup.end

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (float*, i32)* @_Z2k1Pfi to i8*))
  br label %setup.end

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void
}

declare dso_local i32 @cudaSetupArgument(i8*, i64, i64) local_unnamed_addr

declare dso_local i32 @cudaLaunch(i8*) local_unnamed_addr

; Function Attrs: uwtable
define dso_local void @_Z11launches_k1Pf(float* %a) local_unnamed_addr #0 {
entry:
  %a.addr.i88 = alloca float*, align 8
  %b.addr.i89 = alloca i32, align 4
  %a.addr.i77 = alloca float*, align 8
  %b.addr.i78 = alloca i32, align 4
  %a.addr.i66 = alloca float*, align 8
  %b.addr.i67 = alloca i32, align 4
  %a.addr.i55 = alloca float*, align 8
  %b.addr.i56 = alloca i32, align 4
  %a.addr.i = alloca float*, align 8
  %b.addr.i = alloca i32, align 4
  %call = tail call i32 @cudaConfigureCall(i64 4294967304, i32 1, i64 4294967304, i32 1, i64 0, %struct.CUstream_st* null)
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %kcall.configok, label %kcall.end

kcall.configok:                                   ; preds = %entry
  %0 = bitcast float** %a.addr.i55 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0)
  %1 = bitcast i32* %b.addr.i56 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %1)
  store float* %a, float** %a.addr.i55, align 8, !tbaa !2
  store i32 4, i32* %b.addr.i56, align 4, !tbaa !6
  %2 = call i32 @cudaSetupArgument(i8* nonnull %0, i64 8, i64 0)
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %setup.next.i57, label %_Z2k1Pfi.exit59

setup.next.i57:                                   ; preds = %kcall.configok
  %4 = call i32 @cudaSetupArgument(i8* nonnull %1, i64 4, i64 8)
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %setup.next1.i58, label %_Z2k1Pfi.exit59

setup.next1.i58:                                  ; preds = %setup.next.i57
  %6 = call i32 @cudaLaunch(i8* bitcast (void (float*, i32)* @_Z2k1Pfi to i8*))
  br label %_Z2k1Pfi.exit59

_Z2k1Pfi.exit59:                                  ; preds = %kcall.configok, %setup.next.i57, %setup.next1.i58
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %1)
  br label %kcall.end

kcall.end:                                        ; preds = %entry, %_Z2k1Pfi.exit59
  %call4 = call i32 @_Z1fv()
  %agg.tmp3.sroa.0.0.insert.ext = zext i32 %call4 to i64
  %agg.tmp3.sroa.0.0.insert.insert = or i64 %agg.tmp3.sroa.0.0.insert.ext, 4294967296
  %call5 = call i32 @cudaConfigureCall(i64 4294967304, i32 1, i64 %agg.tmp3.sroa.0.0.insert.insert, i32 1, i64 0, %struct.CUstream_st* null)
  %tobool6 = icmp eq i32 %call5, 0
  br i1 %tobool6, label %kcall.configok7, label %kcall.end8

kcall.configok7:                                  ; preds = %kcall.end
  %7 = bitcast float** %a.addr.i66 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %7)
  %8 = bitcast i32* %b.addr.i67 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %8)
  store float* %a, float** %a.addr.i66, align 8, !tbaa !2
  store i32 4, i32* %b.addr.i67, align 4, !tbaa !6
  %9 = call i32 @cudaSetupArgument(i8* nonnull %7, i64 8, i64 0)
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %setup.next.i68, label %_Z2k1Pfi.exit70

setup.next.i68:                                   ; preds = %kcall.configok7
  %11 = call i32 @cudaSetupArgument(i8* nonnull %8, i64 4, i64 8)
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %setup.next1.i69, label %_Z2k1Pfi.exit70

setup.next1.i69:                                  ; preds = %setup.next.i68
  %13 = call i32 @cudaLaunch(i8* bitcast (void (float*, i32)* @_Z2k1Pfi to i8*))
  br label %_Z2k1Pfi.exit70

_Z2k1Pfi.exit70:                                  ; preds = %kcall.configok7, %setup.next.i68, %setup.next1.i69
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %7)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %8)
  br label %kcall.end8

kcall.end8:                                       ; preds = %kcall.end, %_Z2k1Pfi.exit70
  %call10 = call i32 @_Z1fv()
  %call12 = call i32 @_Z1fv()
  %agg.tmp9.sroa.0.0.insert.ext = zext i32 %call10 to i64
  %agg.tmp9.sroa.0.0.insert.insert = or i64 %agg.tmp9.sroa.0.0.insert.ext, 4294967296
  %agg.tmp11.sroa.0.0.insert.ext = zext i32 %call12 to i64
  %agg.tmp11.sroa.0.0.insert.insert = or i64 %agg.tmp11.sroa.0.0.insert.ext, 4294967296
  %call13 = call i32 @cudaConfigureCall(i64 %agg.tmp9.sroa.0.0.insert.insert, i32 1, i64 %agg.tmp11.sroa.0.0.insert.insert, i32 1, i64 0, %struct.CUstream_st* null)
  %tobool14 = icmp eq i32 %call13, 0
  br i1 %tobool14, label %kcall.configok15, label %kcall.end16

kcall.configok15:                                 ; preds = %kcall.end8
  %14 = bitcast float** %a.addr.i77 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %14)
  %15 = bitcast i32* %b.addr.i78 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %15)
  store float* %a, float** %a.addr.i77, align 8, !tbaa !2
  store i32 4, i32* %b.addr.i78, align 4, !tbaa !6
  %16 = call i32 @cudaSetupArgument(i8* nonnull %14, i64 8, i64 0)
  %17 = icmp eq i32 %16, 0
  br i1 %17, label %setup.next.i79, label %_Z2k1Pfi.exit81

setup.next.i79:                                   ; preds = %kcall.configok15
  %18 = call i32 @cudaSetupArgument(i8* nonnull %15, i64 4, i64 8)
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %setup.next1.i80, label %_Z2k1Pfi.exit81

setup.next1.i80:                                  ; preds = %setup.next.i79
  %20 = call i32 @cudaLaunch(i8* bitcast (void (float*, i32)* @_Z2k1Pfi to i8*))
  br label %_Z2k1Pfi.exit81

_Z2k1Pfi.exit81:                                  ; preds = %kcall.configok15, %setup.next.i79, %setup.next1.i80
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %14)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %15)
  br label %kcall.end16

kcall.end16:                                      ; preds = %kcall.end8, %_Z2k1Pfi.exit81
  %call18 = call i32 @_Z1fv()
  %call20 = call i32 @_Z1fv()
  %agg.tmp17.sroa.0.0.insert.ext = zext i32 %call18 to i64
  %agg.tmp17.sroa.0.0.insert.insert = or i64 %agg.tmp17.sroa.0.0.insert.ext, 4294967296
  %agg.tmp19.sroa.0.0.insert.ext = zext i32 %call20 to i64
  %agg.tmp19.sroa.0.0.insert.insert = or i64 %agg.tmp19.sroa.0.0.insert.ext, 4294967296
  %call21 = call i32 @cudaConfigureCall(i64 %agg.tmp17.sroa.0.0.insert.insert, i32 1, i64 %agg.tmp19.sroa.0.0.insert.insert, i32 1, i64 0, %struct.CUstream_st* null)
  %tobool22 = icmp eq i32 %call21, 0
  br i1 %tobool22, label %kcall.configok23, label %kcall.end26

kcall.configok23:                                 ; preds = %kcall.end16
  %call24 = call float* @_Z1gv()
  %call25 = call i32 @_Z1fv()
  %21 = bitcast float** %a.addr.i88 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %21)
  %22 = bitcast i32* %b.addr.i89 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %22)
  store float* %call24, float** %a.addr.i88, align 8, !tbaa !2
  store i32 %call25, i32* %b.addr.i89, align 4, !tbaa !6
  %23 = call i32 @cudaSetupArgument(i8* nonnull %21, i64 8, i64 0)
  %24 = icmp eq i32 %23, 0
  br i1 %24, label %setup.next.i90, label %_Z2k1Pfi.exit92

setup.next.i90:                                   ; preds = %kcall.configok23
  %25 = call i32 @cudaSetupArgument(i8* nonnull %22, i64 4, i64 8)
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %setup.next1.i91, label %_Z2k1Pfi.exit92

setup.next1.i91:                                  ; preds = %setup.next.i90
  %27 = call i32 @cudaLaunch(i8* bitcast (void (float*, i32)* @_Z2k1Pfi to i8*))
  br label %_Z2k1Pfi.exit92

_Z2k1Pfi.exit92:                                  ; preds = %kcall.configok23, %setup.next.i90, %setup.next1.i91
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %21)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %22)
  br label %kcall.end26

kcall.end26:                                      ; preds = %kcall.end16, %_Z2k1Pfi.exit92
  %call28 = call i32 @_Z1fv()
  %call29 = call i32 @_Z1fv()
  %add = add nsw i32 %call29, %call28
  %call31 = call i32 @_Z1fv()
  %mul = shl nsw i32 %call31, 1
  %agg.tmp27.sroa.0.0.insert.ext = zext i32 %add to i64
  %agg.tmp27.sroa.0.0.insert.insert = or i64 %agg.tmp27.sroa.0.0.insert.ext, 4294967296
  %agg.tmp30.sroa.0.0.insert.ext = zext i32 %mul to i64
  %agg.tmp30.sroa.0.0.insert.insert = or i64 %agg.tmp30.sroa.0.0.insert.ext, 4294967296
  %call32 = call i32 @cudaConfigureCall(i64 %agg.tmp27.sroa.0.0.insert.insert, i32 1, i64 %agg.tmp30.sroa.0.0.insert.insert, i32 1, i64 0, %struct.CUstream_st* null)
  %tobool33 = icmp eq i32 %call32, 0
  br i1 %tobool33, label %kcall.configok34, label %kcall.end39

kcall.configok34:                                 ; preds = %kcall.end26
  %call35 = call float* @_Z1gv()
  %call36 = call i32 @_Z1fv()
  %call37 = call i32 @_Z1fv()
  %add38 = add nsw i32 %call37, %call36
  %28 = bitcast float** %a.addr.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %28)
  %29 = bitcast i32* %b.addr.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %29)
  store float* %call35, float** %a.addr.i, align 8, !tbaa !2
  store i32 %add38, i32* %b.addr.i, align 4, !tbaa !6
  %30 = call i32 @cudaSetupArgument(i8* nonnull %28, i64 8, i64 0)
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %setup.next.i, label %_Z2k1Pfi.exit

setup.next.i:                                     ; preds = %kcall.configok34
  %32 = call i32 @cudaSetupArgument(i8* nonnull %29, i64 4, i64 8)
  %33 = icmp eq i32 %32, 0
  br i1 %33, label %setup.next1.i, label %_Z2k1Pfi.exit

setup.next1.i:                                    ; preds = %setup.next.i
  %34 = call i32 @cudaLaunch(i8* bitcast (void (float*, i32)* @_Z2k1Pfi to i8*))
  br label %_Z2k1Pfi.exit

_Z2k1Pfi.exit:                                    ; preds = %kcall.configok34, %setup.next.i, %setup.next1.i
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %28)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %29)
  br label %kcall.end39

kcall.end39:                                      ; preds = %kcall.end26, %_Z2k1Pfi.exit
  ret void
}

declare dso_local i32 @cudaConfigureCall(i64, i32, i64, i32, i64, %struct.CUstream_st*) local_unnamed_addr #1

declare dso_local i32 @_Z1fv() local_unnamed_addr #1

declare dso_local float* @_Z1gv() local_unnamed_addr #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }

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
