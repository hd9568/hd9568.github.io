---
title: 'CUDA Streams：异步执行、Event 同步与计算传输重叠'
description: '理解 CUDA Stream 的异步执行模型，掌握 Event 同步、cudaMemcpyAsync 和 Ping-Pong Buffer 实现计算与数据传输重叠。'
category: 'CUDA'
pubDate: '2026-06-16'
updatedDate: '2026-06-16'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [Stream 是什么](#二stream-是什么)
3. [默认流与异步执行](#三默认流与异步执行)
4. [Event 同步](#四event-同步)
5. [计算与数据传输重叠](#五计算与数据传输重叠)
6. [Ping-Pong Buffer](#六ping-pong-buffer)
7. [常见坑](#七常见坑)
8. [面试回答模板](#八面试回答模板)
9. [总结](#九总结)

## 一、核心结论

CUDA Stream 用来组织 GPU 上的异步任务队列。

- 同一个 Stream 内任务按提交顺序执行。
- 不同 Stream 中的任务在资源允许时可以并发。
- Kernel launch 默认是异步的，Host 不会等待 kernel 完成。
- `cudaEvent_t` 可以记录某个 Stream 的进度，并让其他 Stream 等待。
- 使用 pinned memory 和 `cudaMemcpyAsync` 才更容易实现 H2D/D2H 与 kernel 的重叠。
- Ping-Pong Buffer 用两套缓冲交替传输和计算，隐藏数据传输延迟。

## 二、Stream 是什么

Stream 是 GPU 任务队列。

```text
Stream 0: H2D copy -> Kernel -> D2H copy
Stream 1: H2D copy -> Kernel -> D2H copy
```

同一个 Stream 内保持顺序；不同 Stream 之间没有默认顺序依赖。

创建 Stream：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

kernel<<<grid, block, 0, stream>>>(args);

cudaStreamDestroy(stream);
```

Kernel 启动语法中第 4 个参数就是 Stream。

## 三、默认流与异步执行

Kernel launch 对 Host 异步：

```cpp
my_kernel<<<grid, block>>>(x);
// Host 线程会继续往下执行，不会自动等 kernel 完成。
```

如果要等待 GPU 完成：

```cpp
cudaDeviceSynchronize();
```

更细粒度等待某个 Stream：

```cpp
cudaStreamSynchronize(stream);
```

使用异步拷贝：

```cpp
cudaMemcpyAsync(d_x, h_x, bytes, cudaMemcpyHostToDevice, stream);
```

注意：Host 内存最好是 pinned memory，否则异步拷贝和重叠效果可能受限。

```cpp
float* h_x = nullptr;
cudaHostAlloc(&h_x, bytes, cudaHostAllocDefault);  // pinned host memory
```

## 四、Event 同步

Event 可以记录一个 Stream 的执行位置。

```cpp
cudaEvent_t event;
cudaEventCreate(&event);

// 在 stream_a 中记录事件。
cudaEventRecord(event, stream_a);

// stream_b 等待 event 完成后再继续。
cudaStreamWaitEvent(stream_b, event, 0);

cudaEventDestroy(event);
```

典型用途：

- 测量 GPU 时间。
- 建立不同 Stream 之间的依赖。
- 避免全局 `cudaDeviceSynchronize()` 带来的过度同步。

测量时间：

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);
float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
```

## 五、计算与数据传输重叠

理想流水线：

```text
Time --->
Stream 0: H2D chunk0 -> Kernel chunk0 -> D2H chunk0
Stream 1:             H2D chunk1 -> Kernel chunk1 -> D2H chunk1
```

当 GPU 支持 copy engine 和 compute engine 并发时，传输和计算可以重叠。

基本条件：

- 使用不同 Stream。
- 使用 `cudaMemcpyAsync`。
- Host 内存是 pinned memory。
- Kernel 计算时间和拷贝时间都有足够规模。
- 硬件支持并发 copy/compute。

## 六、Ping-Pong Buffer

Ping-Pong Buffer 使用两套 device buffer 交替处理数据块。

```cpp
constexpr int kStreams = 2;
cudaStream_t streams[kStreams];
float* d_in[kStreams];
float* d_out[kStreams];

for (int i = 0; i < kStreams; ++i) {
    cudaStreamCreate(&streams[i]);
    cudaMalloc(&d_in[i], chunk_bytes);
    cudaMalloc(&d_out[i], chunk_bytes);
}

for (int chunk = 0; chunk < num_chunks; ++chunk) {
    int s = chunk % kStreams;

    // 第 s 套 buffer 处理当前 chunk。
    cudaMemcpyAsync(d_in[s], h_in + chunk * chunk_elems,
                    chunk_bytes, cudaMemcpyHostToDevice, streams[s]);

    // 同一个 stream 内，kernel 会等当前 H2D 完成。
    kernel<<<grid, block, 0, streams[s]>>>(d_in[s], d_out[s], chunk_elems);

    cudaMemcpyAsync(h_out + chunk * chunk_elems, d_out[s],
                    chunk_bytes, cudaMemcpyDeviceToHost, streams[s]);
}

for (int i = 0; i < kStreams; ++i) {
    cudaStreamSynchronize(streams[i]);
}
```

注释里的关键点：

- 同一 Stream 内顺序保证，不需要在 H2D 和 Kernel 之间手动同步。
- 不同 Stream 之间可重叠执行。
- 两套 buffer 避免下一块数据覆盖上一块还没用完的 buffer。

## 七、常见坑

### 1. 用 pageable memory 期待重叠

普通 `malloc/new` 的 Host 内存不一定能真正异步传输。使用 pinned memory 更稳。

### 2. 过度 cudaDeviceSynchronize

```cpp
cudaMemcpyAsync(..., stream);
cudaDeviceSynchronize();  // 过度同步，破坏流水线
kernel<<<..., stream>>>(...);
```

应尽量用 Stream 或 Event 做局部同步。

### 3. Chunk 太小

Chunk 太小会导致 launch overhead 和拷贝开销占比过高，重叠效果不明显。

### 4. 多 Stream 不一定更快

如果 kernel 已经占满全部 SM，或者拷贝引擎已满，多 Stream 可能没有收益。

## 八、面试回答模板

如果问题是“CUDA Stream 如何实现重叠”，可以这样回答：

1. Stream 是 GPU 异步任务队列，同一 Stream 顺序执行，不同 Stream 可并发。
2. Kernel launch 和 `cudaMemcpyAsync` 都可以异步提交。
3. 要实现 H2D/D2H 与 kernel 重叠，通常需要不同 Stream、pinned memory、异步拷贝和硬件支持。
4. Event 用于记录 Stream 进度，也可以让一个 Stream 等待另一个 Stream 的事件。
5. Ping-Pong Buffer 用两套缓冲交替传输和计算，减少数据传输对总时间的影响。

## 九、总结

Stream 优化的目标是把串行流程改成流水线：

```text
传输 chunk i+1 的同时，计算 chunk i
```

它适合数据分块明显、传输成本可观、计算和拷贝可并发的场景。实际收益必须用 Nsight Systems 看时间线验证。
