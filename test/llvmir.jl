using DebuggerFramework
using Cxx
using Iterators
reload("DebuggerFramework")
module LLVMIRDebugger
  using DebuggerFramework
  using Cxx
  using Iterators

  Cxx.addHeaderDir("/home/keno/julia/deps/srccache/llvm-3.9.1/")
  cxx"""
  #include "llvm/IR/Function.h"
  #include "llvm/IR/Module.h"
  #include "llvm/IR/BasicBlock.h"
  #include "llvm/IRReader/IRReader.h"
  #include "llvm/Support/MemoryBuffer.h"
  #include "llvm/IR/ModuleSlotTracker.h"

  // This is a private LLVM header at the moment
  #include "lib/ExecutionEngine/Interpreter/Interpreter.h"

  extern llvm::LLVMContext jl_LLVMContext;
  """

  # Utils
  if !isdefined(:StartAtIterator)
      struct StartAtIterator{T,S}
          it::T
          state::S
      end
      Base.start(it::StartAtIterator) = it.state
      Base.next(it::StartAtIterator, state) = Base.next(it.it, state)
      Base.done(it::StartAtIterator, state) = Base.done(it.it, state)
      startat(it, state) = StartAtIterator(it, state)
  end

  # The interpreter
  struct ExecutionContext
      M::pcpp"llvm::Module"
      Interp::pcpp"llvm::Interpreter"
      ExecutionContext(M::pcpp"llvm::Module") = new(M,
          icxx"new llvm::Interpreter(std::unique_ptr<llvm::Module>{$M});")
  end
  CurBB(ctx::ExecutionContext) = icxx"$(ctx.Interp)->ECStack.back().CurBB;"
  CurInst(ctx::ExecutionContext) = icxx"$(ctx.Interp)->ECStack.back().CurInst;"
  CurFunction(ctx::ExecutionContext) = icxx"$(ctx.Interp)->ECStack.back().CurFunction;"

  function Base.convert(::Type{vcpp"llvm::MemoryBufferRef"}, data::String)
      icxx"""
        llvm::MemoryBufferRef{StringRef{$(pointer(data)),(size_t)$(sizeof(data))},
                        "converted"};
      """
  end

  function Base.parse(::Type{pcpp"llvm::Module"}, data)
      mod = icxx"""
      llvm::SMDiagnostic Err;
      auto mod =
        llvm::parseIR($(convert(vcpp"llvm::MemoryBufferRef", data)),
                      Err, jl_LLVMContext);
      if (!mod) {
        Err.print("llvmir.jl", llvm::errs());
      }
      mod.release();
      """
      if mod == C_NULL
          error("Parsing failed")
      end
      mod
  end

  function Base.print(io::IO, Val::pcpp"llvm::Value")
      print(io, unsafe_string(icxx"""
          std::string str;
          llvm::raw_string_ostream OS(str);
          $Val->print(OS, true);
          OS.str();
          str;
      """))
  end

  Base.start(BB::pcpp"llvm::BasicBlock") = icxx"$BB->begin();"
  Base.done(BB::pcpp"llvm::BasicBlock", i) = icxx"$i == $BB->end();"
  Base.next(BB::pcpp"llvm::BasicBlock", i) = (icxx"*$i;", icxx"++$i;")
  prev(BB::pcpp"llvm::BasicBlock", i) = (icxx"*$i;", icxx"--$i;")

  function step_one!(state::ExecutionContext)
      # Don't use the acessor, because we also want to increment
      icxx"$(state.Interp)->visit(*$(state.Interp)->ECStack.back().CurInst++);"
  end

  # Debugger stuff
  function enter(F::pcpp"llvm::Function")
      ec = ExecutionContext(icxx"$F->getParent();")
      icxx"$(ec.Interp)->callFunction($F, llvm::ArrayRef<llvm::GenericValue>());"
      ec
  end

  # Debugger support
  struct LLVMIRFrame <: DebuggerFramework.StackFrame
      state::ExecutionContext
      idx::Int
  end
  CurFunction(frame::LLVMIRFrame) = icxx"$(frame.state.Interp)->ECStack[$(frame.idx)].CurFunction;"

  function DebuggerFramework.debug(F::pcpp"llvm::Function", args...)
      state = enter(F)
      stack = [LLVMIRFrame(state, 0)]
      DebuggerFramework.RunDebugger(stack, args...)
  end

  function getName(V::Union{pcpp"llvm::BasicBlock", pcpp"llvm::Function", pcpp"llvm::Value"})
      unsafe_string(icxx"$(V)->getName();")
  end

  function DebuggerFramework.print_status_synthtic(io::IO, state, frame::LLVMIRFrame, lines_before, total_lines)
      es = frame.state
      cur_it = icxx"auto it = $(CurInst(es)); it;" # Fake copy constructor
      show_bb_name = false
      for i = 1:lines_before
          if icxx"&*$(cur_it) == &$(CurBB(es))->front();"
              show_bb_name = true
              break
          end
          cur_it = prev(CurBB(es), cur_it)[2]
      end
      active_line = cur_line = 0
      if show_bb_name
          println(io, getName(CurBB(es)),":")
          cur_line += 1
          total_lines -= 1
      end
      for inst in Base.Iterators.take(startat(CurBB(es), cur_it), 5)
          cur_line += 1
          if icxx"&$inst == &*$(CurInst(es));"
              active_line = cur_line
          end
          println(io, icxx"(llvm::Value*)&$inst;")
      end
      active_line
  end

  function DebuggerFramework.locdesc(frame::LLVMIRFrame)
      F = CurFunction(frame)
      BB = icxx"$(frame.state.Interp)->ECStack[$(frame.idx)].CurBB;"
      "function $(getName(F)), BB $(getName(BB))"
  end

  struct AggregateValue
      values::Vector{Any}
  end
  Base.print(io::IO, val::AggregateValue) = print(io, val.values)

  function fromGenericValue(typ::pcpp"llvm::Type", val::vcpp"llvm::GenericValue")
      if icxx"$typ->isVoidTy();"
          return nothing
      elseif icxx"$typ->isHalfTy() || $typ->isX86_FP80Ty() ||
                  $typ->isFP128Ty() || $typ->isPPC_FP128Ty();"
          error("Unsupported floating point type")
      elseif icxx"$typ->isFloatTy();"
          return icxx"$val.FloatVal;"
      elseif icxx"$typ->isDoubleTy();"
          return icxx"$val.DoubleVal;"
      elseif icxx"$typ->isPointerTy();"
          return icxx"$val.PointerVal;"
      elseif icxx"$typ->isAggregateType();"
          return AggregateValue(Any[fromGenericValue(x...) for x in
            zip(icxx"$typ->subtypes();", icxx"$val.AggregateVal;")])
      elseif icxx"$typ->isIntegerTy();"
          # Good enough for now
          return parse(BigInt, unsafe_string(icxx"$val.IntVal.toString(10, false);"))
      else
          error("Unsupported type")
      end
  end

  function DebuggerFramework.print_locals(io::IO, frame::LLVMIRFrame)
      MST = icxx"new llvm::ModuleSlotTracker{$(frame.state.M), true};"
      icxx"$MST->incorporateFunction(*$(CurFunction(frame)));"
      for (CompileTimeVal, RunTimeVal) in icxx"$(frame.state.Interp)->ECStack[$(frame.idx)].Values;"
          name = string("%",icxx"$CompileTimeVal->hasName();" ?
            getName(CompileTimeVal) :
            string(icxx"$MST->getLocalSlot($CompileTimeVal);"))
          DebuggerFramework.print_var(io, name, Nullable(
              fromGenericValue(icxx"$CompileTimeVal->getType();", RunTimeVal)
            ), nothing)
      end
      icxx"delete $MST;"
  end

  function update_stack!(ds, state)
      stacksize = Int(icxx"$(state.Interp)->ECStack.size();"-1)
      ds.stack = [LLVMIRFrame(state, i) for i = stacksize:-1:0]
      if ds.level > length(ds.stack)
          ds.level = length(ds.stack)
      end
  end

  function DebuggerFramework.execute_command(state, frame::LLVMIRFrame, cmd::Val{:si}, command)
      step_one!(frame.state)
      update_stack!(state, frame.state)
      return true
  end
end

ir = """
@.str = private unnamed_addr constant [13 x i8] c"hello world\\0A\\00"
declare i32 @puts(i8* nocapture) nounwind
define void @julia_foo(i8*, i8*) {
top:
  br label %if

if:                                               ; preds = %top, %pass
  %"#temp#.02" = phi i64 [ 1, %top ], [ %4, %if ]
  %2 = and i64 %"#temp#.02", 255
  %3 = icmp eq i64 %2, %"#temp#.02"
  %4 = add nuw nsw i64 %"#temp#.02", 1
  %5 = trunc i64 %"#temp#.02" to i8
  %6 = add nsw i64 %"#temp#.02", -1
  %7 = getelementptr i8, i8* %1, i64 %6
  %8 = load i8, i8* %7, align 1
  %9 = zext i8 %8 to i64
  %10 = add nsw i64 %9, -1
  %11 = getelementptr i8, i8* %0, i64 %10
  store i8 %5, i8* %11, align 1
  %12 = icmp eq i64 %4, 21
  br i1 %12, label %done, label %if

done:
  ret void
}
define i32 @main() {   ; i32()*
top:
    %temp = alloca [13 x i8]
    %casttemp = getelementptr [13 x i8], [13 x i8]* %temp, i64 0, i64 0
    %cast210 = getelementptr [13 x i8], [13 x i8]* @.str, i64 0, i64 0
    call void @julia_foo(i8* %casttemp, i8* %cast210)
    call i32 @puts(i8* %cast210)
    ret i32 0
}
"""
M = parse(pcpp"llvm::Module", ir)
F = icxx"""$M->getFunction("main");"""
