#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <iostream>
#include <vector>

@interface MetalKernel : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLLibrary> library;
@property (nonatomic, strong) id<MTLFunction> kernelFunction;
@property (nonatomic, strong) id<MTLComputePipelineState> pipelineState;
- (instancetype)init;
- (void)compute:(const float*)input weights:(const float*)weights output:(float*)output size:(int)size;
@end

@implementation MetalKernel

- (instancetype)init {
    self = [super init];
    if (self) {
        // Create Metal device and command queue
        self.device = MTLCreateSystemDefaultDevice();
        self.commandQueue = [self.device newCommandQueue];
        
        // Load the Metal shader library
        NSError* error = nil;
        self.library = [self.device newDefaultLibrary];
        if (!self.library) {
            std::cerr << "Failed to create Metal library" << std::endl;
            return nil;
        }
        
        // Get the kernel function
        self.kernelFunction = [self.library newFunctionWithName:@"transformerKernel"];
        if (!self.kernelFunction) {
            std::cerr << "Failed to create kernel function" << std::endl;
            return nil;
        }
        
        // Create compute pipeline state
        self.pipelineState = [self.device newComputePipelineStateWithFunction:self.kernelFunction error:&error];
        if (!self.pipelineState) {
            std::cerr << "Failed to create pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
            return nil;
        }
    }
    return self;
}

- (void)compute:(const float*)input weights:(const float*)weights output:(float*)output size:(int)size {
    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [self.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Create buffers
    id<MTLBuffer> inputBuffer = [self.device newBufferWithBytes:input
                                                      length:size * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> weightsBuffer = [self.device newBufferWithBytes:weights
                                                        length:size * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [self.device newBufferWithLength:size * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    
    // Set pipeline state and buffers
    [encoder setComputePipelineState:self.pipelineState];
    [encoder setBuffer:inputBuffer offset:0 atIndex:0];
    [encoder setBuffer:weightsBuffer offset:0 atIndex:1];
    [encoder setBuffer:outputBuffer offset:0 atIndex:2];
    
    // Dispatch threads
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(std::min(size, 256), 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    // End encoding and commit
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy results back
    memcpy(output, [outputBuffer contents], size * sizeof(float));
    
    // Release resources
    [inputBuffer release];
    [weightsBuffer release];
    [outputBuffer release];
}

@end

// Global Metal kernel instance
static MetalKernel* metalKernel = nil;

void apply_softmax(float* scores, int size) {
    if (!metalKernel) {
        metalKernel = [[MetalKernel alloc] init];
    }
    
    // Use Metal for softmax computation
    std::vector<float> temp(size);
    [metalKernel compute:scores weights:nullptr output:temp.data() size:size];
    memcpy(scores, temp.data(), size * sizeof(float));
}