package com.example.android.camera2basic;

import android.content.res.AssetFileDescriptor;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.util.Log;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.experimental.GpuDelegate;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class TfliteInferer implements AutoCloseable {
    private static final int INPUT_SIZE = 360;
    private static final int NUMBER_OF_BODY_PARTS = 19;
    private static final int NUMBER_OF_PAF_COMPONENTS = 38;
    private static final int BYTES_PER_CHANNEL = 4;
    private static final String TAG = "TfliteInferer";

    private final Interpreter interpreter;
    private final Delegate delegate;
    private final ByteBuffer inputBuffer;
    private final Map<Integer, Object> outputs;

    public TfliteInferer(Resources resources, boolean useDelegate) throws IOException {
        Log.i(TAG, "Initializing on thread " + Thread.currentThread());

        Interpreter.Options interpreterOptions = new Interpreter.Options();
        if (useDelegate) {
            delegate = new GpuDelegate();
            interpreterOptions.addDelegate(delegate);
        } else {
            delegate = null;
        }

        interpreter = new Interpreter(readModelFromResource(resources, R.raw.pose_model), interpreterOptions);

        int outputSize = INPUT_SIZE / 8;
        float[][][][] heatmap = new float[1][outputSize][outputSize][NUMBER_OF_BODY_PARTS];
        float[][][][] heatmapMaxpool = new float[1][outputSize][outputSize][NUMBER_OF_BODY_PARTS];
        float[][][][] pafTensor = new float[1][outputSize][outputSize][NUMBER_OF_PAF_COMPONENTS];

        outputs = new HashMap<>();
        outputs.put(0, heatmapMaxpool);
        outputs.put(1, heatmap);
        outputs.put(2, pafTensor);

        inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * BYTES_PER_CHANNEL);
        inputBuffer.order(ByteOrder.nativeOrder());
    }

    private static ByteBuffer readModelFromResource(Resources resources, int modelResourceId) throws IOException {
        AssetFileDescriptor modelFd = resources.openRawResourceFd(modelResourceId);
        FileChannel fileChannel = modelFd.createInputStream().getChannel();
        long startOffset = modelFd.getStartOffset();
        long declaredLength = modelFd.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    public void close() throws IOException{
        interpreter.close();
        if (delegate instanceof Closeable) {
            ((Closeable) delegate).close();
        }
    }

    public void detectForFrame(Bitmap bitmap) {
        Log.i(TAG, "Running inference on thread " + Thread.currentThread());

        writeToInput(bitmap);
        interpreter.runForMultipleInputsOutputs(new ByteBuffer[]{inputBuffer}, outputs);
        readFromOutputs();
    }

    private void writeToInput(Bitmap bitmap) {
        Bitmap rescaled = rescaleInput(bitmap);
        int[] flatInputPixels = new int[INPUT_SIZE * INPUT_SIZE];
        rescaled.getPixels(flatInputPixels, 0, rescaled.getWidth(), 0, 0, rescaled.getWidth(), rescaled.getHeight());
        for (int i = 0; i < flatInputPixels.length; i++) {
            flatInputPixels[i] = 0;
        }
        inputBuffer.rewind();
        for (int pixelValue : flatInputPixels) {
            inputBuffer.putFloat(((float) ((pixelValue >> 16) & 0xFF)));
            inputBuffer.putFloat(((float) ((pixelValue >> 8) & 0xFF)));
            inputBuffer.putFloat(((float) (pixelValue & 0xFF)));
        }
    }

    private Bitmap rescaleInput(Bitmap bitmap) {
        Bitmap rescaled = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(rescaled);
        Paint paint = new Paint();
        paint.setColor(Color.BLACK);
        Matrix matrix = new Matrix();
        matrix.postTranslate(bitmap.getWidth() / 2f, bitmap.getHeight() / 2f);
        matrix.postScale(INPUT_SIZE / (float) bitmap.getWidth(), INPUT_SIZE / (float) bitmap.getHeight());
        matrix.postTranslate(-INPUT_SIZE / 2f, -INPUT_SIZE / 2f);
        canvas.drawBitmap(bitmap, matrix, paint);
        return rescaled;
    }

    private void readFromOutputs() {
        double eps = 0.0001;
        Map<String, Double> results = new HashMap<>();
        results.put("heatmapMaxpool", zeroRatio(((float[][][][]) outputs.get(0))[0], eps));
        results.put("heatmap", zeroRatio(((float[][][][]) outputs.get(1))[0], eps));
        results.put("heatmapZeros", zeroValues(((float[][][][]) outputs.get(1))[0]));
        results.put("pafTensor", zeroRatio(((float[][][][]) outputs.get(2))[0], eps));
        Log.i(TAG, "Results: " + results);
    }


    private static double zeroRatio(float[][][] floats, double eps) {
        int count = 0;
        int total = 0;
        for (int i = 0; i < floats.length; i++) {
            for (int j = 0; j < floats[i].length; j++) {
                for (int k = 0; k < floats[i][j].length; k++) {
                    total++;
                    if (Math.abs(floats[i][j][k]) < eps) {
                        count++;
                    }
                }
            }
        }
        if (total > 0) {
            return count / (double) total;
        } else {
            return Double.NaN;
        }
    }

    private static double zeroValues(float[][][] floats) {
        int count = 0;
        int total = 0;
        for (int i = 0; i < floats.length; i++) {
            for (int j = 0; j < floats[i].length; j++) {
                for (int k = 0; k < floats[i][j].length; k++) {
                    total++;
                    if (Math.abs(floats[i][j][k]) == 0.0f) {
                        count++;
                    }
                }
            }
        }
        if (total > 0) {
            return count / (double) total;
        } else {
            return Double.NaN;
        }
    }

}
