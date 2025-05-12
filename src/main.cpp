#include <Arduino.h>        // required in .cpp for Arduino projects
#include <TinyMLShield.h>   // your camera shield library
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h" //could get an error
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "open_hand_model_data.h"
#include <string.h>
/*
  Active Learning Labs
  Harvard University 
  tinyMLx - OV7675 Camera Test
*/

static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;

static const int kTensorArenaSize = 100 * 1024; // 100KB
static uint8_t tensor_arena[kTensorArenaSize];

// pointers to the model and interpreter
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;


// flags for command handling
bool commandRecv = false;
bool liveFlag    = false;
bool captureFlag = false;

// QCIF buffer: 176×144 @ RGB565 (2 bytes/pixel)
uint8_t image[176 * 144 * 2];
int bytesPerFrame;



void initModel() {
  // 1) Map the model

  Serial.println("Loading model...");
  const tflite::Model* model = tflite::GetModel(open_hand_model_data);

  /*
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema v%d not v%d",
                            model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }  */


  Serial.println("building op resolver...");
  // 2) This pulls in all the ops we'll need (Conv2D, Pooling, etc.)
  static tflite::AllOpsResolver resolver;
  Serial.println("Buidling the interpreter...");
  // 3) Build the interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  Serial.println("allocating tensors...");
  // 4) Allocate memory from the tensor_arena for the tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  
  Serial.print("AllocateTensors status: ");
  Serial.println(alloc_status);  
  if (alloc_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    Serial.println("AllocateTensors() failed");
    return;
  }
  Serial.println("obtaining pointers to input/output tensors...");
  // 5) Obtain pointers to the in/out tensors
  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);
  if (input_tensor == nullptr || output_tensor == nullptr) {
    error_reporter->Report("Failed to get input/output tensors");
    Serial.println("Failed to get input/output tensors");
    return;
  }

  Serial.print("In  tensor: scale="); Serial.print(input_tensor->params.scale, 6);
  Serial.print(", zp=");            Serial.println(input_tensor->params.zero_point);
  Serial.print("Out tensor: scale="); Serial.print(output_tensor->params.scale, 6);
  Serial.print(", zp=");             Serial.println(output_tensor->params.zero_point);
  Serial.println("initmodel() done");

  error_reporter->Report("Model initialized OK");
}



void setup() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println("Starting up...");
  initModel();             // ← this sets up interpreter + tensors
  initializeShield();
  

  // Initialize the OV7675 camera
  if (!Camera.begin(QCIF, GRAYSCALE, 1, OV7675)) {
    Serial.println("Failed to initialize camera");
    while (1);
  }



  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();
  Serial.println("Camera initialized. ");
  /*
  Serial.println("Welcome to the OV7675 test\n");
  Serial.println("Available commands:\n");
  Serial.println("single  - take a single image and print hexadecimal pixels (default)");
  Serial.println("live    - stream raw image bytes continuously");
  Serial.println("capture - in single mode, triggers a capture"); */
}

// ───────────────────────── CONFIG ─────────────────────────
const uint16_t SRC_W = 176, SRC_H = 144;   // camera settings
const uint16_t CROP_W = 96,  CROP_H = 96;  // final frame size

// (Put the two frame buffers in global / static RAM on small MCUs)
uint8_t fullFrame[SRC_W * SRC_H];          // 25 344  B
uint8_t cropped[CROP_W * CROP_H];          // 9 216   B

unsigned long t0;
unsigned long t1;
unsigned long t2;
unsigned long t3;
unsigned long t_total;   // microseconds from t0 to t3
unsigned long t_infer;   // microseconds from t2 to t3

void loop() {
  t0 = micros();  // start time
  //testing something here
  
  Camera.readFrame(image);
  const int srcW = Camera.width();   // 176
  const int srcH = Camera.height();  // 144


  uint8_t downsampled[96 * 96];

  // 3) Nearest-neighbor downsample from 176×144 → 96×96
  //const int srcW = Camera.width();   // 176
  //const int srcH = Camera.height();  // 144
  for (int y = 0; y < 96; y++) {
    int src_y = y * srcH / 96;      // maps 0→0, 95→143
    for (int x = 0; x < 96; x++) {
      int src_x = x * srcW / 96;    // maps 0→0, 95→175
      int idx   = (src_y * srcW + src_x);  // 2 bytes/pixel
      downsampled[y * 96 + x] = image[idx];   // take the Y byte
    }
  }
  // Serial.println("FRAME_START");
  // // send 96*96 bytes of raw gray data:
  // Serial.write(downsampled, 96*96);
  // // flush in case of buffering:
  // Serial.println(); // a newline after the block
  // Serial.println("FRAME_END");



  // bool stop_program = true;
  // while(stop_program) {}
  
  //done testing

  t1 = micros();  // end of camera read
  

  



  // Serial.print("Downsampled pix: ");
  // for (int i = 0; i < 5; i++) {
  //   Serial.print(downsampled[i]);
  //   Serial.print(' ');
  // }
  // Serial.println();

    // 4) Copy into the TFLM input tensor
  //memcpy(input_tensor->data.uint8, downsampled, 96 * 96); // had to remove this and put in the loop below
  // for (int i = 0; i < 96*96; i++) {
  //   int8_t q = int8_t(downsampled[i] - 128);
  //   input_tensor->data.int8[i] = q;
  // }

  float in_scale =  input_tensor->params.scale;
  int   in_zp    =  input_tensor->params.zero_point;

  for (int i = 0; i < 96*96; i++) {
    // normalize raw 0–255 pixel to [0,1]
    float normalized = downsampled[i] / 255.0f;

    // quantize: q = round(normalized / scale) + zero_point
    int32_t q = (int32_t)round(normalized / in_scale) + in_zp;

    // clamp to int8 range:
    q = q <  -128 ? -128 : (q > 127 ? 127 : q);

    input_tensor->data.int8[i] = (int8_t)q;
  }


  // //testing something here
  // int8_t min_q =  127, max_q = -128;
  // for (int i = 0; i < 96*96; i++) {
  //   int8_t v = input_tensor->data.int8[i];
  //   min_q = v < min_q ? v : min_q;
  //   max_q = v > max_q ? v : max_q;
  // }
  // Serial.print("Input Q range: [");
  // Serial.print(min_q);
  // Serial.print(", ");
  // Serial.print(max_q);
  // Serial.println("]");


  t2 = micros();  // end of camera read

  // 5) Invoke the model
  interpreter->Invoke();
  int8_t raw_q = output_tensor->data.int8[0];
  //Serial.print("Raw q output: "); Serial.println(raw_q);

  // 6) Read the output tensor
  // uint8_t raw_score = output_tensor->data.uint8[0];
  // float confidence = (raw_score +128)/ 255.0f;

  float out_scale =  output_tensor->params.scale;
  int   out_zp    =  output_tensor->params.zero_point;

  //int8_t raw_q    = output_tensor->data.int8[0];
  float confidence = (raw_q - out_zp) * out_scale;

  if (confidence < 0.46) {
    Serial.println("Hand detected! Confidence: " + String((1-confidence) * 100.0, 2) + "%");
  } else {
    Serial.println("No hand detected. Confidence: " + String(confidence * 100.0, 2) + "%");
  }
  t3 = micros();  // end of inference
  t_total = t3 - t0;  // total time for the loop
  t_infer = t3 - t2;  // inference time only
  Serial.print("Inference time: ");
  Serial.print(t_infer); Serial.print(" us, ");
  Serial.print("Total time: ");
  Serial.print(t_total); Serial.println(" us");


  // (optional) Print the first few quantized inputs too:
  // Serial.print("First inputs: ");
  // for (int i = 0; i < 5; i++) {
  //   Serial.print(input_tensor->data.int8[i]); 
  //   Serial.print(' ');
  // }
  // Serial.println();
  // Serial.println();




  delay(500);
  




/*
  // Read serial commands
  String cmd;
  while (Serial.available()) {
    char c = Serial.read();
    if (c != '\n' && c != '\r') cmd += c;
    else if (c == '\r') {
      commandRecv = true;
      cmd.toLowerCase();
    }
  }

  if (commandRecv) {
    commandRecv = false;
    if (cmd == "live") {
      liveFlag = true;
      Serial.println("Entering live stream in 5s...");
      delay(5000);
    } 
    else if (cmd == "single") {
      liveFlag = false;
      Serial.println("Single mode: type \"capture\" to snap.");
      delay(200);
    } 
    else if (cmd == "capture" && !liveFlag) {
      captureFlag = true;
      delay(200);
    }
  }

  // Live streaming mode
  if (liveFlag) {
    Camera.readFrame(image);
    Serial.write(image, bytesPerFrame);
    delay(100);  // adjust as needed
    return;
  }

  // Single capture mode
  if (captureFlag) {
    captureFlag = false;
    Camera.readFrame(image);
    Serial.println("Printing image hex in 3s...");
    delay(3000);
    for (int i = 0; i < bytesPerFrame; i += 2) {
      // print high-byte then low-byte so pixel is in correct order
      Serial.print("0x");
      Serial.print(image[i+1], HEX);
      Serial.print(image[i],   HEX);
      if (i + 2 < bytesPerFrame) Serial.print(", ");
    }
    Serial.println();
    delay(500);
  }
    */
}