/*
 * Copyright 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.android.camera2basic;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.v7.app.AppCompatActivity;
import android.widget.Toast;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.Semaphore;

public class CameraActivity extends AppCompatActivity implements ImageReader.OnImageAvailableListener {

    public static final boolean USE_DELEGATE = false;
    private Semaphore isProcessingFrame = new Semaphore(1);
    private HandlerThread handlerThread;
    private Handler handler;
    private TfliteInferer tfliteInferer;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        if (null == savedInstanceState) {
            getSupportFragmentManager().beginTransaction()
                    .replace(R.id.container, Camera2BasicFragment.newInstance(this))
                    .commit();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        this.handlerThread = new HandlerThread("inference");
        handlerThread.start();
        this.handler = new Handler(handlerThread.getLooper());
        handler.post(() -> {
            // create in same thread where inference happens
            try {
                tfliteInferer = new TfliteInferer(getResources(), USE_DELEGATE);
            } catch (IOException e) {
                Toast.makeText(this, "Exception if TfliteInferer: " + e, Toast.LENGTH_LONG).show();
            }
        });
    }

    @Override
    protected void onPause() {
        super.onPause();
        this.handlerThread.quitSafely();
        this.handlerThread = null;
        this.handler = null;
        tfliteInferer.close();
        tfliteInferer = null;
    }

    @Override
    public void onImageAvailable(ImageReader reader) {
        final Bitmap bitmap;
        try (final Image image = reader.acquireLatestImage()) {
            if (image == null) {
                return;
            }

            if (!isProcessingFrame.tryAcquire()) {
                return;
            }

            ByteBuffer imagePlane = image.getPlanes()[0].getBuffer();
            byte[] buffer = new byte[imagePlane.capacity()];
            imagePlane.get(buffer);
            bitmap = BitmapFactory.decodeByteArray(buffer, 0, buffer.length);
        }

        handler.post(() -> {
            tfliteInferer.detectForFrame(bitmap);
            isProcessingFrame.release();
        });


    }
}
