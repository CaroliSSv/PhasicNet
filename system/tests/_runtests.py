import sys, traceback
def run(mod):
    names=[n for n in dir(mod) if n.startswith("test_")]; p=f=0
    for n in names:
        try: getattr(mod,n)(); p+=1; print(f"  PASS {n}")
        except Exception as e: f+=1; print(f"  FAIL {n}: {type(e).__name__}: {str(e)[:120]}")
    print(f"{mod.__name__}: {p} passed, {f} failed"); return f
