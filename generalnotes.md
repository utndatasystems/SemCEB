very good work.

I left a couple of comments and opened a PR. no need to merge the PR, it's just for highlighting the diff.

please also add a .gitignore

`semantic-selectivity run` didn't work for me out of the box because `weasyprint` wants to load some library on my system but doesn't find it. Error is as follows:
```

WeasyPrint could not import some external libraries. Please carefully follow the installation steps before reporting an issue:
https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation
https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#troubleshooting 

-----

Traceback (most recent call last):
  File "/Users/agij47es/Documents/UTN_Research/semantic-selectivity/semantic-selectivity-estimation/.venv/bin/semantic-selectivity", line 5, in <module>
    from run import main
  File "/Users/agij47es/Documents/UTN_Research/semantic-selectivity/semantic-selectivity-estimation/run.py", line 9, in <module>
    from results.plotter import ResultsPlotter
  File "/Users/agij47es/Documents/UTN_Research/semantic-selectivity/semantic-selectivity-estimation/results/plotter.py", line 12, in <module>
    from weasyprint import HTML
  File "/Users/agij47es/Documents/UTN_Research/semantic-selectivity/semantic-selectivity-estimation/.venv/lib/python3.12/site-packages/weasyprint/__init__.py", line 372, in <module>
    from .css import preprocess_stylesheet  # noqa: I001, E402
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/agij47es/Documents/UTN_Research/semantic-selectivity/semantic-selectivity-estimation/.venv/lib/python3.12/site-packages/weasyprint/css/__init__.py", line 29, in <module>
    from ..text.fonts import FontConfiguration
  File "/Users/agij47es/Documents/UTN_Research/semantic-selectivity/semantic-selectivity-estimation/.venv/lib/python3.12/site-packages/weasyprint/text/fonts.py", line 17, in <module>
    from .constants import (  # isort:skip
  File "/Users/agij47es/Documents/UTN_Research/semantic-selectivity/semantic-selectivity-estimation/.venv/lib/python3.12/site-packages/weasyprint/text/constants.py", line 5, in <module>
    from .ffi import pango
  File "/Users/agij47es/Documents/UTN_Research/semantic-selectivity/semantic-selectivity-estimation/.venv/lib/python3.12/site-packages/weasyprint/text/ffi.py", line 476, in <module>
    gobject = _dlopen(
              ^^^^^^^^
  File "/Users/agij47es/Documents/UTN_Research/semantic-selectivity/semantic-selectivity-estimation/.venv/lib/python3.12/site-packages/weasyprint/text/ffi.py", line 464, in _dlopen
    return ffi.dlopen(names[0], flags)  # pragma: no cover
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/agij47es/Documents/UTN_Research/semantic-selectivity/semantic-selectivity-estimation/.venv/lib/python3.12/site-packages/cffi/api.py", line 150, in dlopen
    lib, function_cache = _make_ffi_library(self, name, flags)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/agij47es/Documents/UTN_Research/semantic-selectivity/semantic-selectivity-estimation/.venv/lib/python3.12/site-packages/cffi/api.py", line 834, in _make_ffi_library
    backendlib = _load_backend_lib(backend, libname, flags)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/agij47es/Documents/UTN_Research/semantic-selectivity/semantic-selectivity-estimation/.venv/lib/python3.12/site-packages/cffi/api.py", line 829, in _load_backend_lib
    raise OSError(msg)
OSError: cannot load library 'libgobject-2.0-0': dlopen(libgobject-2.0-0, 0x0002): tried: 'libgobject-2.0-0' (no such file), '/System/Volumes/Preboot/Cryptexes/OSlibgobject-2.0-0' (no such file), '/usr/lib/libgobject-2.0-0' (no such file, not in dyld cache), 'libgobject-2.0-0' (no such file), '/opt/local/lib/libgobject-2.0-0' (no such file), '/libgobject-2.0-0' (no such file).  Additionally, ctypes.util.find_library() did not manage to locate a library called 'libgobject-2.0-0'
```

It seems to be related to this issue: https://github.com/Kozea/WeasyPrint/issues/2427
Adding DYLD_FALLBACK_LIBRARY_PATH=/opt/local/lib:$DYLD_FALLBACK_LIBRARY_PATH also didn't fix it for me. I see you only use it for table rendering. Is there maybe a different library that we can use for the same thing?
