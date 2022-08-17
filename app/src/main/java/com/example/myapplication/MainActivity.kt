package com.example.myapplication

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import com.google.firebase.ml.modeldownloader.CustomModelDownloadConditions
import com.google.firebase.ml.modeldownloader.DownloadType
import com.google.firebase.ml.modeldownloader.FirebaseModelDownloader
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    var interpreter : Interpreter ?= null
    var selectedPhotoUri : Uri? = null
    var bitmap : Bitmap?= null


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)



        val conditions = CustomModelDownloadConditions.Builder()
            .requireWifi()
            .build()
        FirebaseModelDownloader.getInstance()
            .getModel("Brain-Tumor", DownloadType.LOCAL_MODEL_UPDATE_IN_BACKGROUND, conditions)
            .addOnCompleteListener {
                // Download complete. Depending on your app, you could enable the ML
                // feature, or switch from the local model to the remote model, etc
                val model = it.result
                val modelFile = model?.file
                if (modelFile != null) {
                    interpreter = Interpreter(modelFile)
                }
            }


        findViewById<Button>(R.id.btn_upload).setOnClickListener {

            Toast.makeText(this, "Select photo", Toast.LENGTH_SHORT).show()
            val i = Intent(Intent.ACTION_PICK)
            i.type= "image/*"
            startActivityForResult(i, 0)
        }


    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(requestCode==0 && resultCode== Activity.RESULT_OK && data!=null){
            Log.i("info", "Photo Selected")
            selectedPhotoUri = data.data
            Log.e("SRI1711", "$selectedPhotoUri")
            //val ibtn_profile = findViewById<de.hdodenhof.circleimageview.CircleImageView>(R.id.ibtn_profile)
            bitmap = MediaStore.Images.Media.getBitmap(
                contentResolver,
                selectedPhotoUri
            )

            test(bitmap!!)
        }
    }


    fun test(bitmap: Bitmap) {
        val bitmap = Bitmap.createScaledBitmap(bitmap, 128, 128, true)
        val input = ByteBuffer.allocateDirect(128*128*3*4).order(ByteOrder.nativeOrder())
        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val px = bitmap.getPixel(x, y)

                // Get channel values from the pixel value.
                val r = Color.red(px)
                val g = Color.green(px)
                val b = Color.blue(px)

                // Normalize channel values to [-1.0, 1.0]. This requirement depends on the model.
                // For example, some models might require values to be normalized to the range
                // [0.0, 1.0] instead.
                val rf = (r - 127) / 255f
                val gf = (g - 127) / 255f
                val bf = (b - 127) / 255f

                input.putFloat(rf)
                input.putFloat(gf)
                input.putFloat(bf)


            }
        }

        val bufferSize = 1000 * java.lang.Float.SIZE / java.lang.Byte.SIZE
        val modelOutput = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder())
        interpreter?.run(input, modelOutput)

        modelOutput.rewind()
        val probabilities = modelOutput.asFloatBuffer()
        Log.i("SRI1711", probabilities.get(0).toString())
        Log.i("SRI1711", probabilities.get(1).toString())
        Log.i("SRI1711", probabilities.get(2).toString())
    }
}