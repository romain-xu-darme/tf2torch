from contextlib import contextmanager
import ctypes
import io
import os, sys
import tempfile

'''Little script to hide stderr messages, in particular those generated by C/C++ that cannot be hidden by redirecting `sys.stderr`.

Usage: 
```
import stderr
f = io.BytesIO()
with stderr_redirector(f):
    do_something_that_may_generate_spurious_stderr_messages
error_messages = f.getvalue().decode('utf-8') # messages sent to stderr are stored here
```

If you don't want to keep the error messages, simply run
```
import stderr
with stderr_redirector():
    do_something_that_may_generate_spurious_stderr_messages
```

Credits: Eli Bendersky
  <https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/>
Edit: Alban Grastien
'''

libc = ctypes.CDLL(None)
c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')

@contextmanager
def stderr_redirector(stream=None):
    # The original fd stderr points to. Usually (?) 2 on POSIX systems.
    original_stderr_fd = sys.stderr.fileno()

    def _redirect_stderr(to_fd):
        """Redirect stderr to the given file descriptor."""
        # Flush the C-level buffer stderr
        libc.fflush(c_stderr)
        # Flush and close sys.stderr - also closes the file descriptor (fd)
        sys.stderr.close()
        # Make original_stderr_fd point to the same file as to_fd
        os.dup2(to_fd, original_stderr_fd)
        # Create a new sys.stderr that points to the redirected fd
        sys.stderr = io.TextIOWrapper(os.fdopen(original_stderr_fd, 'wb'))

    # Save a copy of the original stderr fd in saved_stderr_fd
    saved_stderr_fd = os.dup(original_stderr_fd)
    try:
        # Create a temporary file and redirect stderr to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stderr(tfile.fileno())
        # Yield to caller, then redirect stderr back to the saved fd
        yield
        _redirect_stderr(saved_stderr_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        if stream != None: 
            stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stderr_fd)
