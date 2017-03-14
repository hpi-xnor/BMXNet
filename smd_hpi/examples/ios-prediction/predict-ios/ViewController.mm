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
    
    uint8_t imageData[kWidth*kHeight];
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    CGContextRef contextRef = CGBitmapContextCreate(imageData,
                                                    kWidth,
                                                    kHeight,
                                                    8,
                                                    kWidth,
                                                    colorSpace,
                                                    kCGImageAlphaNone | kCGBitmapByteOrder32Big);
    CGContextDrawImage(contextRef, CGRectMake(0, 0, kWidth, kHeight), image.CGImage);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    NSString *str = @"";
    for (int y = 0; y < kHeight; y++) {
        for (int x = 0; x < kWidth; x++) {
            str = [NSString stringWithFormat: @"%@%@", str, imageData[y * kWidth + x] >= 255 ? @" " : @"#"];
        }
        str =  [str stringByAppendingString:@"\n"];
    }
    
    NSLog(@"%@\n\n", str);
    
    //< copy to float buffer
    std::vector<float> input_buffer(kWidth*kHeight);
    for (int i = 0; i < kHeight; i++) {
        for (int j = 0; j < kWidth; j++) {
            input_buffer.data()[i] = imageData[i];
        }
    }
    
    mx_uint *shape = nil;
    mx_uint shape_len = 0;
    MXPredSetInput(predictor, "data", input_buffer.data(), kWidth*kHeight);
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
    
    return [NSString stringWithFormat: @"%zu (%f)", max_idx, outputs.at(max_idx)]; //[[model_synset objectAtIndex:max_idx] componentsJoinedByString:@" "];
}

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    self.indicatorView = [UIActivityIndicatorView new];
    [self.indicatorView setActivityIndicatorViewStyle:UIActivityIndicatorViewStyleGray];
    
    if (!predictor) {
        NSString *jsonPath = [[NSBundle mainBundle] pathForResource:@"binary-mnist-qall-symbol.json" ofType:nil];
        NSString *paramsPath = [[NSBundle mainBundle] pathForResource:@"binary-mnist-qall-0001.params" ofType:nil];
        model_symbol = [[NSString alloc] initWithData:[[NSFileManager defaultManager] contentsAtPath:jsonPath] encoding:NSUTF8StringEncoding];
        model_params = [[NSFileManager defaultManager] contentsAtPath:paramsPath];
        
        NSString *input_name = @"data";
        const char *input_keys[1];
        input_keys[0] = [input_name UTF8String];
        const mx_uint input_shape_indptr[] = {0, 4};
        const mx_uint input_shape_data[] = {1, 1, kWidth, kHeight};
        MXPredCreate([model_symbol UTF8String], [model_params bytes], (int)[model_params length], 1, 0, 1,
                     input_keys, input_shape_indptr, input_shape_data, &predictor);
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
                //self.labelDescription.text = [self predictImage:self.imageViewPhoto.image];
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
    //UIImage * grayScaledImg = [self grayImage:sourceImage];
    
    GPUImageAdaptiveThresholdFilter *stillImageFilter = [[GPUImageAdaptiveThresholdFilter alloc] init];
    stillImageFilter.blurRadiusInPixels = 8.0;
    UIImage *retImage = [stillImageFilter imageByFilteringImage:sourceImage];
    
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
    double x = CGImageGetHeight(cgImage)/2.0 - cropRectSize/2.0;
    double y = CGImageGetWidth(cgImage)/2.0 - cropRectSize/2.0;
    CGRect cropRect = CGRectMake(x, y, cropRectSize, cropRectSize);
    CGContextSetLineWidth(context, 5);
    CGContextSetStrokeColorWithColor(context, [[ UIColor redColor ] CGColor]);
    CGContextStrokeRect(context, cropRect);
    UIImage* augmentedImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    // crop and threshold image
    UIImage *thresholded = [self doBinarize: [self cropCenterRect:augmentedImage toSize: cropRectSize]];
    
    // visualize 28x28 pic that will go into neural net
    UIGraphicsBeginImageContextWithOptions(CGSizeMake(kWidth, kHeight), NO, 1.0);
    CGContextRef context2 = UIGraphicsGetCurrentContext();
    UIGraphicsPushContext(context2);
    [thresholded drawInRect:CGRectMake(0, 0, kWidth, kHeight)];
    UIGraphicsPopContext();
    UIImage* newImage2 = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    // classify 28x28 pic
    NSString *classification = [self classifyNumber:thresholded];
    
    // update ui
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(){
        dispatch_async(dispatch_get_main_queue(), ^(){
            [self.imageViewPhoto setImage: augmentedImage];
            [self.imageViewCrop setImage: newImage2];
            self.labelDescription.text = classification;
            //[self.imageViewPhoto setImage: newImage]; //[UIImage imageWithCGImage: newImage.CGImage scale:0.0 orientation:UIImageOrientationRightMirrored]];
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
