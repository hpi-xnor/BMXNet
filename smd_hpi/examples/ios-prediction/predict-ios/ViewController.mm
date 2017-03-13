//
//  ViewController.m
//  WhatIsThis
//
//  Created by Haoxiang Li on 1/23/16.
//  Copyright © 2016 Haoxiang Li. All rights reserved.
//

#import "ViewController.h"
#import <AVFoundation/AVFoundation.h>
#import <AVFoundation/AVCaptureDevice.h> // For access to the camera
#import <AVFoundation/AVCaptureInput.h> // For adding a data input to the camera
#import <AVFoundation/AVCaptureSession.h>
#import "GPUImage.h"
#import "GPUImageAdaptiveThresholdFilter.h"

@interface ViewController () <UIImagePickerControllerDelegate, UINavigationControllerDelegate>

@property (nonatomic, retain) UIActivityIndicatorView *indicatorView;

@end

@implementation ViewController


- (NSString *)classifyNumber:(UIImage *)image {
    NSDate *methodStart = [NSDate date];
    
    const int width = 28;
    const int height = 28;
    uint8_t imageData[width*height];
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    CGContextRef contextRef = CGBitmapContextCreate(imageData,
                                                    width,
                                                    height,
                                                    8,
                                                    width,
                                                    colorSpace,
                                                    kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    CGContextDrawImage(contextRef, CGRectMake(0, 0, width, height), image.CGImage);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    //< copy to float buffer
    std::vector<float> input_buffer(width*height);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            input_buffer.data()[i] = imageData[i];
        }
    }
    
    mx_uint *shape = nil;
    mx_uint shape_len = 0;
    MXPredSetInput(predictor, "data", input_buffer.data(), width*height);
    MXPredForward(predictor);
    MXPredGetOutputShape(predictor, 0, &shape, &shape_len);
    mx_uint tt_size = 1;
    for (mx_uint i = 0; i < shape_len; i++) {
        tt_size *= shape[i];
    }
    std::vector<float> outputs(tt_size);
    MXPredGetOutput(predictor, 0, outputs.data(), tt_size);
    size_t max_idx = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
    
    NSDate *methodFinish = [NSDate date];
    NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
    NSLog(@"executionTime = %f", executionTime);
    
    return [[model_synset objectAtIndex:max_idx] componentsJoinedByString:@" "];
}


- (NSString *)predictImage:(UIImage *)image {
    NSDate *methodStart = [NSDate date];

    const int numForRendering = kDefaultWidth*kDefaultHeight*(kDefaultChannels+1);
    const int numForComputing = kDefaultWidth*kDefaultHeight*kDefaultChannels;

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    uint8_t imageData[numForRendering];
    CGContextRef contextRef = CGBitmapContextCreate(imageData,
                                                    kDefaultWidth,
                                                    kDefaultHeight,
                                                    8,
                                                    kDefaultWidth*(kDefaultChannels+1),
                                                    colorSpace,
                                                    kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    CGContextDrawImage(contextRef, CGRectMake(0, 0, kDefaultWidth, kDefaultHeight), image.CGImage);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);

    //< Subtract the mean and copy to the input buffer
    std::vector<float> input_buffer(numForComputing);
    float *p_input_buffer[3] = {
        input_buffer.data(),
        input_buffer.data() + kDefaultWidth*kDefaultHeight,
        input_buffer.data() + kDefaultWidth*kDefaultHeight*2};
    const float *p_mean[3] = {
        model_mean,
        model_mean + kDefaultWidth*kDefaultHeight,
        model_mean + kDefaultWidth*kDefaultHeight*2};
    for (int i = 0, map_idx = 0, glb_idx = 0; i < kDefaultHeight; i++) {
        for (int j = 0; j < kDefaultWidth; j++) {
            p_input_buffer[0][map_idx] = imageData[glb_idx++] - p_mean[0][map_idx];
            p_input_buffer[1][map_idx] = imageData[glb_idx++] - p_mean[1][map_idx];
            p_input_buffer[2][map_idx] = imageData[glb_idx++] - p_mean[2][map_idx];
            glb_idx++;
            map_idx++;
        }
    }
    
    mx_uint *shape = nil;
    mx_uint shape_len = 0;
    MXPredSetInput(predictor, "data", input_buffer.data(), numForComputing);
    MXPredForward(predictor);
    MXPredGetOutputShape(predictor, 0, &shape, &shape_len);
    mx_uint tt_size = 1;
    for (mx_uint i = 0; i < shape_len; i++) {
        tt_size *= shape[i];
    }
    std::vector<float> outputs(tt_size);
    MXPredGetOutput(predictor, 0, outputs.data(), tt_size);
    size_t max_idx = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
    
    NSDate *methodFinish = [NSDate date];
    NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
    NSLog(@"executionTime = %f", executionTime);
    
    return [[model_synset objectAtIndex:max_idx] componentsJoinedByString:@" "];
}

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    self.indicatorView = [UIActivityIndicatorView new];
    [self.indicatorView setActivityIndicatorViewStyle:UIActivityIndicatorViewStyleGray];
    
    if (!predictor) {
        NSString *jsonPath = [[NSBundle mainBundle] pathForResource:@"Inception_BN-symbol.json" ofType:nil];
        NSString *paramsPath = [[NSBundle mainBundle] pathForResource:@"Inception_BN-0039.params" ofType:nil];
        NSString *meanPath = [[NSBundle mainBundle] pathForResource:@"mean_224.bin" ofType:nil];
        NSString *synsetPath = [[NSBundle mainBundle] pathForResource:@"synset.txt" ofType:nil];
        NSLog(@"%@", meanPath);
        model_symbol = [[NSString alloc] initWithData:[[NSFileManager defaultManager] contentsAtPath:jsonPath] encoding:NSUTF8StringEncoding];
        model_params = [[NSFileManager defaultManager] contentsAtPath:paramsPath];
        
        NSString *input_name = @"data";
        const char *input_keys[1];
        input_keys[0] = [input_name UTF8String];
        const mx_uint input_shape_indptr[] = {0, 4};
        const mx_uint input_shape_data[] = {1, kDefaultChannels, kDefaultWidth, kDefaultHeight};
        MXPredCreate([model_symbol UTF8String], [model_params bytes], (int)[model_params length], 1, 0, 1,
                     input_keys, input_shape_indptr, input_shape_data, &predictor);
        
        NSData *meanData = [[NSFileManager defaultManager] contentsAtPath:meanPath];
        [meanData getBytes:model_mean length:[meanData length]];
        
        model_synset = [NSMutableArray new];
        NSString* synsetText = [NSString stringWithContentsOfFile:synsetPath
                                  encoding:NSUTF8StringEncoding error:nil];
        NSArray* lines = [synsetText componentsSeparatedByCharactersInSet:
                                    [NSCharacterSet newlineCharacterSet]];
        for (NSString *l in lines) {
            NSArray *parts = [l componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
            if ([parts count] > 1) {
                [model_synset addObject:[parts subarrayWithRange:NSMakeRange(1, [parts count]-1)]];
            }
        }
        
        /*//< Visualize the Mean Data
        std::vector<uint8_t> mean_with_alpha(kDefaultWidth*kDefaultHeight*(kDefaultChannels+1), 0);
        float *p_mean[3] = {
            model_mean,
            model_mean + kDefaultWidth*kDefaultHeight,
            model_mean + kDefaultWidth*kDefaultHeight*2};
        for (int i = 0, map_idx = 0, glb_idx = 0; i < kDefaultHeight; i++) {
            for (int j = 0; j < kDefaultWidth; j++) {
                mean_with_alpha[glb_idx++] = p_mean[0][map_idx];
                mean_with_alpha[glb_idx++] = p_mean[1][map_idx];
                mean_with_alpha[glb_idx++] = p_mean[2][map_idx];
                mean_with_alpha[glb_idx++] = 0;
                map_idx++;
            }
        }
        
        NSData *mean_data = [NSData dataWithBytes:mean_with_alpha.data() length:mean_with_alpha.size()*sizeof(float)];
        CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)mean_data);
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        // Creating CGImage from cv::Mat
        CGImageRef imageRef = CGImageCreate(kDefaultWidth,
                                            kDefaultHeight,
                                            8,
                                            8*(kDefaultChannels+1),
                                            kDefaultWidth*(kDefaultChannels+1),
                                            colorSpace,
                                            kCGImageAlphaNone|kCGBitmapByteOrderDefault,
                                            provider,
                                            NULL,
                                            false,
                                            kCGRenderingIntentDefault
                                            );
        meanImage = [UIImage imageWithCGImage:imageRef];
        CGImageRelease(imageRef);
        CGDataProviderRelease(provider);
        self.imageViewPhoto.image = meanImage;*/
    }
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
    NSLog(@"Received Memory Warning!");
}

- (IBAction)selectPhotoButtonTapped:(id)sender {
    UIImagePickerController *imagePicker = [UIImagePickerController new];
    imagePicker.allowsEditing = NO;
    imagePicker.sourceType =  UIImagePickerControllerSourceTypePhotoLibrary;
    imagePicker.delegate = self;
    [self presentViewController:imagePicker animated:YES completion:nil];
}

- (IBAction)capturePhotoButtonTapped:(id)sender {
    UIImagePickerController *imagePicker = [UIImagePickerController new];
    imagePicker.allowsEditing = NO;
    imagePicker.sourceType =  UIImagePickerControllerSourceTypeCamera;
    imagePicker.delegate = self;
    [self presentViewController:imagePicker animated:YES completion:nil];
}

- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<NSString *,id> *)info {
    UIImage *chosenImage = info[UIImagePickerControllerOriginalImage];
    self.imageViewPhoto.image = chosenImage;
    [picker dismissViewControllerAnimated:YES completion:^(void){
        [self.view addSubview:self.indicatorView];
        self.indicatorView.frame = self.view.bounds;
        [self.indicatorView startAnimating];
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(){
            dispatch_async(dispatch_get_main_queue(), ^(){
                self.labelDescription.text = [self predictImage:self.imageViewPhoto.image];
                [self.indicatorView stopAnimating];
            });
        });
    }];
}

- (void)imagePickerControllerDidCancel:(UIImagePickerController *)picker {
    [picker dismissViewControllerAnimated:YES completion:nil];
}

- (IBAction)startDetectionButtonTapped:(id)sender {
    [self.detectionButton setTitle:@"Stop Detection" forState:UIControlStateNormal];
    [self.detectionButton addTarget:self
                             action:@selector(stopDetectionButtonTapped:)
                   forControlEvents:UIControlEventTouchUpInside];
    if (!captureSession) {
        captureSession = [self createCaptureSession];
    }
    [captureSession startRunning];
}

- (IBAction)stopDetectionButtonTapped:(id)sender {
    [self.detectionButton setTitle:@"Start Detection" forState:UIControlStateNormal];
    [self.detectionButton addTarget:self
                             action:@selector(startDetectionButtonTapped:)
                   forControlEvents:UIControlEventTouchUpInside];
    [captureSession stopRunning];
}

- (AVCaptureSession *)createCaptureSession
{
    AVCaptureSession *session = [[AVCaptureSession alloc] init];
    session.sessionPreset = AVCaptureSessionPresetHigh;
    
    AVCaptureDevice *device = [self selectCameraAt:AVCaptureDevicePositionBack];
    
    
    NSError *error = nil;
    AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
    if (!input) {
        // Handle the error appropriately.
        NSLog(@"no input.....");
    }
    [session addInput:input];
    
    AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
    [session addOutput:output];
    output.videoSettings = @{ (NSString *)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA) };
    
    dispatch_queue_t queue = dispatch_queue_create("MyQueue", NULL);
    
    [output setSampleBufferDelegate:self queue:queue];
    
    return session;
}

- (AVCaptureDevice *)selectCameraAt:(AVCaptureDevicePosition)chosenPosition {
    NSArray *devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    for (AVCaptureDevice *device in devices) {
        if ([device position] == chosenPosition) {
            return device;
        }
    }
    return nil;
}

- (UIImage *) cropCenterRect:(UIImage *)image toSize:(int)size
{
    double x = image.size.width/2.0 - size/2.0;
    double y = image.size.height/2.0 - size/2.0;

    CGRect cropRect = CGRectMake(x, y, size, size);
    CGImageRef imageRef = CGImageCreateWithImageInRect([image CGImage], cropRect);
    
    UIImage *cropped = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    
    return cropped;
}

- (UIImage *) doBinarize:(UIImage *)sourceImage
{
    //first off, try to grayscale the image using iOS core Image routine
    UIImage * grayScaledImg = [self grayImage:sourceImage];
    
    GPUImageAdaptiveThresholdFilter *stillImageFilter = [[GPUImageAdaptiveThresholdFilter alloc] init];
    stillImageFilter.blurRadiusInPixels = 8.0;
    UIImage *retImage = [stillImageFilter imageByFilteringImage:grayScaledImg];
    
    return retImage;
}

- (UIImage *) grayImage :(UIImage *)inputImage
{
    // Create a graphic context.
    UIGraphicsBeginImageContextWithOptions(inputImage.size, NO, 1.0);
    CGRect imageRect = CGRectMake(0, 0, inputImage.size.width, inputImage.size.height);
    
    // Draw the image with the luminosity blend mode.
    // On top of a white background, this will give a black and white image.
    [inputImage drawInRect:imageRect blendMode:kCGBlendModeLuminosity alpha:1.0];
    
    // Get the resulting image.
    UIImage *outputImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    return outputImage;
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection {
    
    CGImageRef cgImage = [self imageFromSampleBuffer:sampleBuffer];
    float cropRectSize = 140;
    
    // red box around detection area
    UIGraphicsBeginImageContextWithOptions(CGSizeMake(CGImageGetHeight(cgImage), CGImageGetWidth(cgImage)), NO, 1.0);
    CGContextRef context = UIGraphicsGetCurrentContext();
    
    UIGraphicsPushContext(context);
    [[UIImage imageWithCGImage: cgImage scale:0.0 orientation:UIImageOrientationRight] drawAtPoint:CGPointZero];
    UIGraphicsPopContext();
    
    //CGContextDrawImage(context, CGRectMake(0, 0, CGImageGetWidth(cgImage), CGImageGetHeight(cgImage)), cgImage);
    double x = CGImageGetHeight(cgImage)/2.0 - cropRectSize/2.0;
    double y = CGImageGetWidth(cgImage)/2.0 - cropRectSize/2.0;
    CGRect cropRect = CGRectMake(x, y, cropRectSize, cropRectSize);
    CGContextSetLineWidth(context, 5);
    CGContextSetStrokeColorWithColor(context, [[ UIColor redColor ] CGColor]);
    CGContextStrokeRect(context, cropRect);
    UIImage* newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    
    const int width = 28;
    const int height = 28;
    uint8_t imageData[width*height];
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    CGContextRef contextRef = CGBitmapContextCreate(imageData,
                                                    width,
                                                    height,
                                                    8,
                                                    width,
                                                    colorSpace,
                                                    kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    CGContextDrawImage(contextRef, CGRectMake(0, 0, width, height), newImage.CGImage);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    
    UIImage *thresholded = [self cropCenterRect:newImage toSize: cropRectSize];

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(){
        dispatch_async(dispatch_get_main_queue(), ^(){
            //[self.imageViewPhoto setImage: thresholded];
            [self.imageViewPhoto setImage: newImage]; //[UIImage imageWithCGImage: newImage.CGImage scale:0.0 orientation:UIImageOrientationRightMirrored]];
            CGImageRelease( cgImage );
        });
    });
}

- (CGImageRef) imageFromSampleBuffer:(CMSampleBufferRef) sampleBuffer
{
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVPixelBufferLockBaseAddress(imageBuffer,0);
    uint8_t *baseAddress = (uint8_t *)CVPixelBufferGetBaseAddressOfPlane(imageBuffer, 0);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    CGContextRef newContext = CGBitmapContextCreate(baseAddress, width, height, 8, bytesPerRow, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
    CGImageRef newImage = CGBitmapContextCreateImage(newContext);
    CGContextRelease(newContext);
    
    CGColorSpaceRelease(colorSpace);
    CVPixelBufferUnlockBaseAddress(imageBuffer,0);
    
    return newImage;
}



@end
