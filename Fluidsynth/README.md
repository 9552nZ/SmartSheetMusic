Some comments to install Fluidsynth and have it work with pretty_midi.
(this is only approximate/partial and should be used in conjunction with the main help page
https://sourceforge.net/p/fluidsynth/wiki/BuildingWithCMake/)

- Download CMake
- Follow the procedure as indicated in "Building on Windows" and "Building with VisualStudio on Windows".
- Make sure to download Visual Studio 2010 (even if not available on the Windows website and even if it weights 6go)
- Make sure to get all the requirements from gtk.org (all the glib and others are required indeed)
- Make sure to unzip the above libraries in C:\freesw, respecting the suggested path tree (e.g. use 7zip for the extraction of the files). Add the C:\freesw\bin path to the system path.
- Not sure that we need MinGW or gcc
- Once all the libraries have been installed, run CMake, pointing at fluidsynth-x.y.z (input) and fluidsynth-x.y.z\build for the output. You can configure, generate and build the solutions.
- This output a bunch of files, including one .exe and libfluidsynth_debug.dll (not sure why there is _debug_ ...)
- The libfluidsynth_debug.dll library is the one that should be made visible to the pyFluidSynth binder library. You can then copy and paste the .dll to one of the system path folder. 


