plugins {
    id("com.android.application")
}

android {
    namespace = "com.sightsync.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.sightsync.app"
        minSdk = 24
        targetSdk = 35
        versionCode = 3
        versionName = "0.3.0-phase3"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
    }

    kotlin {
        jvmToolchain(21)
    }

    androidResources {
        noCompress += "tflite"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.activity:activity-ktx:1.9.3")

    implementation("androidx.camera:camera-core:1.3.4")
    implementation("androidx.camera:camera-camera2:1.3.4")
    implementation("androidx.camera:camera-lifecycle:1.3.4")
    implementation("androidx.camera:camera-view:1.3.4")

    // tensorflow-lite and tensorflow-lite-gpu both transitively pull tensorflow-lite-api; letting
    // Gradle resolve it independently via each parent causes a "Cannot access class
    // 'Interpreter.Options'" compile error (a known TFLite+AGP interaction). Pinning it as an
    // explicit direct dependency at the same version forces one unambiguous resolution.
    implementation("org.tensorflow:tensorflow-lite-api:2.16.1")
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
    // tensorflow-lite-gpu's own GpuDelegate.Options extends GpuDelegateFactory.Options, which
    // lives in this separate API artifact -- not bundled in tensorflow-lite-gpu itself, and not
    // pulled in transitively by it either, hence "Cannot access class 'Options'" without this.
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.16.1")

    implementation("com.google.ar:core:1.56.0")
}
