#include "cnn_model_W100_S20.h"

// ===== TensorFlow Lite Micro headers ===== 
#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ===== TFLM interpreter objects ===== 
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// ===== Prediction =====
uint8_t bestClass = 2;  // default fidgeting

// ===== params (match training) =====
const uint8_t SENSORS = 6;
const uint8_t WINDOW  = 100;
const uint8_t STRIDE = 20;
const uint8_t NCLASS  = 7;

// ===== model I/O quantization (from *_io.json) =====
const float IN_SCALE  = 0.003921568859368563;
const int8_t IN_ZERO  = -128;
const float OUT_SCALE = 0.00390625;
const int8_t OUT_ZERO = -128;
float subNorm[SENSORS * 2]; // min-max scalars for each sensor

// ===== sampling =====
const uint32_t SAMPLE_PERIOD = 10000; // 100 Hz (10ms)
uint64_t last_sample_tm = 0;
const uint8_t L_HIP_GPIO = 1;
const uint8_t L_KNEE_GPIO = 2;
const uint8_t L_ANKL_GPIO = 3;
const uint8_t R_HIP_GPIO = 4;
const uint8_t R_KNEE_GPIO = 5;
const uint8_t R_ANKL_GPIO = 6;
constexpr float ADC_SCALE = 3.3 / 4095.0;

// ====== TF LiteMicro arena ======
constexpr size_t kTensorArenaSize = 20 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// ===== Buffers =====
float rawBuf[WINDOW][SENSORS];  // scaled raw value buffer
int rawBufWritePos = 0;
bool rawBufFull = false;

int samplesSeen = 0; // total samples processed
int nextTrigger = WINDOW; // next stride trigger

// ===== MicroROS headers ===== 
#include <micro_ros_arduino.h>
#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>

#include <std_msgs/msg/int32_multi_array.h>

rcl_publisher_t publisher;
std_msgs__msg__Int32MultiArray msg;
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;

// FUNCTIONS
#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){delay(1);}}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){}}

inline int8_t q_int8(float x) {
// Quantize a float x to int8 using your model's input scale & zero-point

  int32_t q = lroundf(x / IN_SCALE + IN_ZERO);
  
  if (q < -128) 
    q = -128; 
  else if (q > 127) 
    q = 127;
  
  return (int8_t)q;
}

void deepsleep() {
//  while(1) {
//    // update with deep sleep code
//  }

  esp_sleep_enable_timer_wakeup(UINT64_MAX);
  esp_deep_sleep_start(); 
  
}

void create_entities() {

  set_microros_wifi_transports("T-3", "@PinkyLove", "192.168.0.102", 8888);
//  set_microros_transports();
  
  allocator = rcl_get_default_allocator();

  //create init_options
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));

  // create node
  RCCHECK(rclc_node_init_default(&node, "current_terrain_node", "", &support));

  // create publisher
  RCCHECK(rclc_publisher_init_best_effort(
    &publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32MultiArray),
    "current_terrain"));

  msg.data.data = (int32_t*)calloc(5, sizeof(int32_t));
  if(!msg.data.data) 
    deepsleep();
  msg.data.size = 5;  
  // 0: timestamp, 1: predicted, 2: input tensor pack time, 3: inf time, 4: pub delay
  
}

void derive_channel_scalars() {
  
  rawBuf[0][0] = ADC_SCALE * analogRead(L_HIP_GPIO); 
  if(rawBuf[0][0] < subNorm[0]) subNorm[0] = rawBuf[0][0];  // L_HIP min
  if(rawBuf[0][0] > subNorm[1]) subNorm[1] = rawBuf[0][0];  // L_HIP max
  
  rawBuf[0][1] = ADC_SCALE * analogRead(L_KNEE_GPIO); 
  if(rawBuf[0][1] < subNorm[2]) subNorm[2] = rawBuf[0][1];  // L_KNEE min
  if(rawBuf[0][1] > subNorm[3]) subNorm[3] = rawBuf[0][1];  // L_KNEE max
  
  rawBuf[0][2] = ADC_SCALE * analogRead(L_ANKL_GPIO); 
  if(rawBuf[0][2] < subNorm[4]) subNorm[4] = rawBuf[0][2];  // L_ANKL min
  if(rawBuf[0][2] > subNorm[5]) subNorm[5] = rawBuf[0][2];  // L_ANKL max
  
  rawBuf[0][3] = ADC_SCALE * analogRead(R_HIP_GPIO);  
  if(rawBuf[0][3] < subNorm[6]) subNorm[6] = rawBuf[0][3];  // R_HIP min
  if(rawBuf[0][3] > subNorm[7]) subNorm[7] = rawBuf[0][3];  // R_HIP max
  
  rawBuf[0][4] = ADC_SCALE * analogRead(R_KNEE_GPIO); 
  if(rawBuf[0][4] < subNorm[8]) subNorm[8] = rawBuf[0][4];  // R_KNEE min
  if(rawBuf[0][4] > subNorm[9]) subNorm[9] = rawBuf[0][4];  // R_KNEE max
  
  rawBuf[0][5] = ADC_SCALE * analogRead(R_ANKL_GPIO); 
  if(rawBuf[0][5] < subNorm[10]) subNorm[10] = rawBuf[0][5];  // R_ANKL min
  if(rawBuf[0][5] > subNorm[11]) subNorm[11] = rawBuf[0][5];  // R_ANKL max
  
}

void fill_input(uint8_t pos) {
  // Raw circular buffer
  rawBuf[pos][0] = (ADC_SCALE * analogRead(L_HIP_GPIO) - subNorm[0]) / (subNorm[1] - subNorm[0]);
  rawBuf[pos][1] = (ADC_SCALE * analogRead(L_KNEE_GPIO) - subNorm[2]) / (subNorm[3] - subNorm[2]);
  rawBuf[pos][2] = (ADC_SCALE * analogRead(L_ANKL_GPIO) - subNorm[4]) / (subNorm[5] - subNorm[4]);
  rawBuf[pos][3] = (ADC_SCALE * analogRead(R_HIP_GPIO) - subNorm[6]) / (subNorm[7] - subNorm[6]);
  rawBuf[pos][4] = (ADC_SCALE * analogRead(R_KNEE_GPIO) - subNorm[8]) / (subNorm[9] - subNorm[8]);
  rawBuf[pos][5] = (ADC_SCALE * analogRead(R_ANKL_GPIO) - subNorm[10]) / (subNorm[11] - subNorm[10]);

}

void setup() {

  Serial.begin(115200);
  delay(2000);

  last_sample_tm = millis();
  for(int i = 0; i< SENSORS*2; i++) {
    if(i%2==0)  subNorm[i] = 3.3; // store min values in even array locations
    else subNorm[i] = 0; // store max values in odd array locations
  }
  while(millis() - last_sample_tm < 120000) // 2 mins for subject to calibrate
    derive_channel_scalars();
  
  create_entities();
  
  // Load model (int8 converted .h file)
  model = tflite::GetModel(cnn_model_W100_S20);
  if (!model) {
    Serial.println("Model pointer null");
    deepsleep();
  }
  Serial.printf("TFLITE version: %d \n", model->version());
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema mismatch. Expect ");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.print(" got ");
    Serial.println(model->version());
    
    deepsleep();
  }
  
  // Resolver:
  static tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();  // GAP
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddExpandDims();
  resolver.AddReshape();
   
  // Interpreter:
  static tflite::MicroInterpreter static_interpreter(
    model, 
    resolver, 
    tensor_arena, 
    kTensorArenaSize
  );

  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed. Increase tensor area."); 
    deepsleep();
  }
  input = interpreter->input(0);    // int8 [1,6,24]
  output = interpreter->output(0);  // int8 [1,9]

  Serial.print("Input bytes: "); Serial.println(input->bytes);
  Serial.print("Input dims: ");
  for (int d = 0; d < input->dims->size; d++) {
    Serial.print(input->dims->data[d]);
    if (d < input->dims->size - 1) Serial.print("x");
  }
  Serial.println();
  Serial.print("Output bytes: "); Serial.println(output->bytes);
  Serial.print("Arena bytes used: "); Serial.println(interpreter->arena_used_bytes());
  Serial.println();
  
  // Initialize buffer positions
  rawBufWritePos = 0;
  rawBufFull = false;
  nextTrigger = WINDOW;
  
  last_sample_tm = micros();
  Serial.println("Deployment mode.\n");

  Serial.println("Setup OK"); delay(100);
}

void loop() {

  uint64_t now = micros();

  // Fixed-rate sampling
  if (now - last_sample_tm >= SAMPLE_PERIOD) {
    last_sample_tm += SAMPLE_PERIOD;

    fill_input(rawBufWritePos);

    rawBufWritePos = (rawBufWritePos + 1) % WINDOW; // circular input buffer idx
    if (!rawBufFull && rawBufWritePos == 0)
      rawBufFull = true;

    samplesSeen++;
  }

  // trigger at stride boundaries
  if (samplesSeen >= nextTrigger && rawBufFull) {

    unsigned long pack_t0 = micros();

    int8_t* in_data = input->data.int8;
    int idx = 0;

    for (int i = 0; i < WINDOW; i++) {
      int buf_idx = (rawBufWritePos + i) % WINDOW;
      for (int s = 0; s < SENSORS; s++) {
        in_data[idx++] = q_int8(rawBuf[buf_idx][s]);
      }
    }

    int in_pack_tm = micros() - pack_t0;

    // Inference
    unsigned long invoke_t0 = micros();

    if (interpreter->Invoke() == kTfLiteOk) {

      int8_t* out_q = output->data.int8;  // [1][NCLASS]
      bestClass = 0;
      int8_t bestScore = out_q[0];

      for (int c = 1; c < NCLASS; c++) {
        if (out_q[c] > bestScore) {
          bestScore = out_q[c];
          bestClass = c;
        }
      }

      int inf_tm = micros() - invoke_t0;

      unsigned long exec_t0 = micros();
      RCSOFTCHECK(rcl_publish(&publisher, &msg, NULL));
      int ros_exec_tm = micros() - exec_t0;

      // Publish
      msg.data.data[0] = millis();
      msg.data.data[1] = (int)bestClass;
      msg.data.data[2] = in_pack_tm;
      msg.data.data[3] = inf_tm;
      msg.data.data[4] = ros_exec_tm;
    }

    nextTrigger += STRIDE;
  }
}
