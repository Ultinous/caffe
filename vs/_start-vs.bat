@echo off

call %~dp0\_env.bat

if errorlevel 1 exit /b %errorlevel%

start buildcaffe.sln
