@REM This script demonstrates the flow of the command through the shell

@REM Don't forget to use the correct python distribution.
SET my_python=C:\ProgramData\Anaconda3\python.exe
SET full_path=%~dp0

@REM Generate templates using chimera.
%my_python% ..\src\Main.py generate templates CHIMERA %fill_path%..\Chimera\GeometricTemplates\ -r 60 -t GEOMETRIC_3D

@REM Generate random tomograms using the generated templates. The criteria used will be [2, 3] and 1 tomogram will be generated.
%my_python% ..\src\Main.py generate tomograms RANDOM ..\Chimera\GeometricTemplates\ ..\Chimera\GeometricTemplates\tomograms.pkl -c 2 3 -n 1

@REM Train an svm on the generated templates and tomograms
%my_python% ..\src\Main.py train my_svm.pkl -t ..\Chimera\GeometricTemplates\ -d ..\Chimera\GeometricTemplates\tomograms.pkl

@REM Generate another tomogram to evaluate. This time the criteria used will be [3, 3]
%my_python% ..\src\Main.py generate tomograms RANDOM ..\Chimera\GeometricTemplates\ ..\Chimera\GeometricTemplates\eval_tomograms.pkl -c 3 3 -n 1

@REM Evaluate the tomogram using the svm
%my_python% ..\src\Main.py eval my_svm.pkl -d ..\Chimera\GeometricTemplates\eval_tomograms.pkl -o my_result.pkl

@REM View results
%my_python% ..\src\Main.py view_results my_result.pkl
