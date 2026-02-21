#include "cnn_model_W100_S20.h"

// SIMULATION DATA
#include "test_sampled_scaled4.cc"

uint16_t sim_i = 0; // index in test_sampled_scaled
uint8_t true_label = 0; // only in simulation
uint8_t bestClass = 4;  // start with 4

// TensorFlow Lite Micro headers
#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ===== model I/O quantization (from *_io.json) =====
const float IN_SCALE  = 0.003921568859368563;
const int8_t IN_ZERO  = -128;
const float OUT_SCALE = 0.00390625;
const int8_t OUT_ZERO = -128;

// ===== params (match training) =====
const uint8_t SENSORS = 6;
const uint8_t WINDOW  = 100;
const uint8_t STRIDE = 20;
const uint8_t NCLASS  = 7;

// ====== TF LiteMicro arena ======
constexpr size_t kTensorArenaSize = 20 * 1024; // constexpr
static uint8_t tensor_arena[kTensorArenaSize];// static

// ===== Buffers =====
float rawBuf[WINDOW][SENSORS];  // Simulation scaled raw value buffer
int rawBufWritePos = 0;
bool rawBufFull = false;

int samplesSeen = 0; // total samples processed
int nextTrigger = WINDOW; // next stride trigger

// TFLM interpreter objects
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Simulation Stats
int confmat[NCLASS][NCLASS];
float total_predictions = 0.0;
long total_correct = 0;

// MicroROS headers
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

/*void timer_callback(rcl_timer_t * timer, int64_t last_call_time) {  
  RCLC_UNUSED(last_call_time);
//  if (timer != NULL) {
//      msg.data = bestClass;
//      msg.data++; // test
      unsigned long ros_t0 = millis();
      RCSOFTCHECK(rcl_publish(&publisher, &msg, NULL));
      int ros_pub_tm = millis() - ros_t0; msg.data.data[5] = ros_pub_tm;
//  }
}
//*/

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
  while(1) {
    // update with deep sleep code
  }
  
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

  // create timer
//  const unsigned int timer_timeout = 1;  // ms
//  RCCHECK(rclc_timer_init_default(
//    &timer,
//    &support,
//    RCL_MS_TO_NS(timer_timeout),
//    timer_callback));

  // create executor
//  RCCHECK(rclc_executor_init_best_effort(&executor, &support.context, 1, &allocator));
//  RCCHECK(rclc_executor_add_timer(&executor, &timer));

  msg.data.data = (int32_t*)calloc(6, sizeof(int32_t));
  if(!msg.data.data) 
    deepsleep();
  msg.data.size = 6;  
  // 0: timestamp, 1: predicted, 2: actual, 3: input tensor pack time, 4: inf time, 5: pub delay
}


void setup() {

  Serial.begin(115200);
  delay(2000);

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
  
  total_predictions = 0;
  total_correct = 0;
  memset(confmat, 0, sizeof(confmat));

  Serial.println("Simulation test mode.\n");

  Serial.println("Setup OK"); delay(100);
}
void loop() {

  // -------- Simulation input --------
  float l_hip  = test_sampled_scaled[sim_i][0];
  float l_knee = test_sampled_scaled[sim_i][1];
  float l_ankl = test_sampled_scaled[sim_i][2];
  float r_hip  = test_sampled_scaled[sim_i][3];
  float r_knee = test_sampled_scaled[sim_i][4];
  float r_ankl = test_sampled_scaled[sim_i][5];

  sim_i = (sim_i + 1) % TEST_SAMPLES; // circular simulation input idx

  // -------- Raw circular buffer --------
  rawBuf[rawBufWritePos][0] = l_hip;
  rawBuf[rawBufWritePos][1] = l_knee;
  rawBuf[rawBufWritePos][2] = l_ankl;
  rawBuf[rawBufWritePos][3] = r_hip;
  rawBuf[rawBufWritePos][4] = r_knee;
  rawBuf[rawBufWritePos][5] = r_ankl;

  rawBufWritePos = (rawBufWritePos + 1) % WINDOW; // circular simulation input buffer idx
  if (!rawBufFull && rawBufWritePos == 0)
    rawBufFull = true;

  samplesSeen++;

  // -------- Trigger at stride boundaries --------
  if (samplesSeen >= nextTrigger && rawBufFull) {

    // ===== Majority label (evaluation only) =====
    
    int last_idx = sim_i - 1;
    if (last_idx < 0) last_idx += TEST_SAMPLES;
    int start_idx = last_idx - (WINDOW - 1);

    int counts_local[NCLASS] = {0};

    if (start_idx >= 0) {
      for (int i = start_idx; i <= last_idx; i++)
        counts_local[(int)test_sampled_scaled[i][SENSORS]]++;
    } else {
      for (int i = start_idx + TEST_SAMPLES; i < TEST_SAMPLES; i++)
        counts_local[(int)test_sampled_scaled[i][SENSORS]]++;
      for (int i = 0; i <= last_idx; i++)
        counts_local[(int)test_sampled_scaled[i][SENSORS]]++;
    }

    true_label = 0;
    for (int c = 1; c < NCLASS; c++)
      if (counts_local[c] > counts_local[true_label])
        true_label = c;
    
    unsigned long pack_t0 = micros();
    // ===== Pack RAW WINDOW -> model input =====
    int8_t* in_data = input->data.int8;   // [1][WINDOW][SENSORS]

    int idx = 0;
    for (int i = 0; i < WINDOW; i++) {
      int buf_idx = (rawBufWritePos + i) % WINDOW;
      for (int s = 0; s < SENSORS; s++) {
        in_data[idx++] = q_int8(rawBuf[buf_idx][s]);
      }
    }

    int in_pack_tm = micros() - pack_t0;

    // ===== Inference =====
    unsigned long invoke_t0 = micros();

    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Invoke failed!");
    } else {
      int8_t* out_q = output->data.int8;  // [1][NCLASS]
      float bestScore = out_q[0];
      bestClass = 0;

      for (int c = 1; c < NCLASS; c++) {
        if (out_q[c] > bestScore) {
          bestScore = out_q[c];
          bestClass = c;
        }
      }

      int inf_tm = micros() - invoke_t0;

      unsigned long exec_t0 = micros();
//      RCSOFTCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(1)));
      RCSOFTCHECK(rcl_publish(&publisher, &msg, NULL));
      int ros_exec_tm = micros() - exec_t0;

      msg.data.data[0] = millis();
      msg.data.data[1] = (int)bestClass;
      msg.data.data[2] = (int)true_label;
      msg.data.data[3] = in_pack_tm;
      msg.data.data[4] = inf_tm;
      msg.data.data[5] = ros_exec_tm;
      
    }

    nextTrigger += STRIDE;
  }
//  RCSOFTCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(1)));
}
