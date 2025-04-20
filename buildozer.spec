[app]

# (str) Title of your application
title = PostureApp

# (str) Package name
package.name = postureapp

# (str) Package domain (needed for android/ios packaging)
package.domain = org.test

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas,tflite,mp3,json

# (str) Application versioning
version = 0.1

# (list) Application requirements
# comma separated e.g. requirements = sqlite3,kivy
requirements = python3,kivy,opencv-python,numpy,tflite-runtime

# (str) Android NDK directory (if empty, it will be automatically downloaded.)
android.ndk_path = /home/alains/Android/Sdk/ndk/android-ndk-r25c

# (str) Android SDK directory (if empty, it will be automatically downloaded.)
android.sdk_path = /home/alains/Android/Sdk

# (list) Permissions
# (See https://python-for-android.readthedocs.io/en/latest/buildoptions/#build-options-1 for all the supported syntaxes and properties)
android.permissions = android.permission.INTERNET,CAMERA,WRITE_EXTERNAL_STORAGE

# (list) The Android archs to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
android.archs = arm64-v8a, armeabi-v7a

[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2
# (str) JAVA binary path - Add this

# Specify the path to JDK 17
p4a.java_build_tool = /usr/lib/jvm/java-17-openjdk-amd64/bin/java

gradle.gradle_options = -Xmx4096m -Dorg.gradle.jvmargs="-Xmx4096m -XX:MaxPermSize=1024m"

android.archs = arm64-v8aandroid.archs = arm64-v8a
