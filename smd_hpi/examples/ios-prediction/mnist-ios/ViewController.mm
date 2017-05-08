//
//  ViewController.m
//  WhatIsThis
//
//  Created by Haoxiang Li on 1/23/16.
//  Copyright © 2016 Haoxiang Li. All rights reserved.
//

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import "ViewController.h"
#import <AVFoundation/AVFoundation.h>
#import <AVFoundation/AVCaptureDevice.h> // For access to the camera
#import <AVFoundation/AVCaptureInput.h> // For adding a data input to the camera
#import <AVFoundation/AVCaptureSession.h>

NSLock *predictionRunningLock = [NSLock new] ;

static void * ExposureTargetBiasContext = &ExposureTargetBiasContext;

@interface ViewController () <UIImagePickerControllerDelegate, UINavigationControllerDelegate>

@property (nonatomic, retain) UIActivityIndicatorView *indicatorView;

@end

@implementation ViewController


- (NSString *)classifyNumber:(UIImage *)image {
    
    uint8_t imageData[kWidth*kHeight];
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    CGContextRef contextRef = CGBitmapContextCreate(imageData,
                                                    kWidth,
                                                    kHeight,
                                                    8,
                                                    kWidth,
                                                    colorSpace,
                                                    kCGImageAlphaNone);
    CGContextDrawImage(contextRef, CGRectMake(0, 0, kWidth, kHeight), image.CGImage);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
//    NSString *str = @"";
//    for (int y = 0; y < kHeight; y++) {
//        for (int x = 0; x < kWidth; x++) {
//            str = [NSString stringWithFormat: @"%@%@", str, imageData[y * kWidth + x] >= 255 ? @" " : @"#"];
//        }
//        str =  [str stringByAppendingString:@"\n"];
//    }
//    
//    NSLog(@"%@\n\n", str);
    
    int hardcoded7[] = {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 84.000000, 185.000000, 159.000000, 151.000000, 60.000000, 36.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 222.000000, 254.000000, 254.000000, 254.000000, 254.000000, 241.000000, 198.000000, 198.000000, 198.000000, 198.000000, 198.000000, 198.000000, 198.000000, 198.000000, 170.000000, 52.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 67.000000, 114.000000, 72.000000, 114.000000, 163.000000, 227.000000, 254.000000, 225.000000, 254.000000, 254.000000, 254.000000, 250.000000, 229.000000, 254.000000, 254.000000, 140.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 17.000000, 66.000000, 14.000000, 67.000000, 67.000000, 67.000000, 59.000000, 21.000000, 236.000000, 254.000000, 106.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 83.000000, 253.000000, 209.000000, 18.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 22.000000, 233.000000, 255.000000, 83.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 129.000000, 254.000000, 238.000000, 44.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 59.000000, 249.000000, 254.000000, 62.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 133.000000, 254.000000, 187.000000, 5.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 9.000000, 205.000000, 248.000000, 58.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 126.000000, 254.000000, 182.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 75.000000, 251.000000, 240.000000, 57.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 19.000000, 221.000000, 254.000000, 166.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 3.000000, 203.000000, 254.000000, 219.000000, 35.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 38.000000, 254.000000, 254.000000, 77.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 31.000000, 224.000000, 254.000000, 115.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 133.000000, 254.000000, 254.000000, 52.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 61.000000, 242.000000, 254.000000, 254.000000, 52.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 121.000000, 254.000000, 254.000000, 219.000000, 40.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 121.000000, 254.000000, 207.000000, 18.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000};
    
    // copy to float buffer
    std::vector<float> input_buffer(kWidth*kHeight);
    for (int i = 0; i < kHeight * kWidth; i++) {
            input_buffer.data()[i] = imageData[i];
    }
    
    
    mx_uint *shape = nil;
    mx_uint shape_len = 0;
    MXPredSetInput(predictor, "data", input_buffer.data(), kWidth*kHeight);
    
    NSDate *methodStart = [NSDate date];
    MXPredForward(predictor);
    NSDate *methodFinish = [NSDate date];
    
    MXPredGetOutputShape(predictor, 0, &shape, &shape_len);
    mx_uint tt_size = 1;
    for (mx_uint i = 0; i < shape_len; i++) {
        tt_size *= shape[i];
    }
    std::vector<float> outputs(tt_size);
    MXPredGetOutput(predictor, 0, outputs.data(), tt_size);
    
    NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
    //NSLog(@"forward pass took %f", executionTime);
    
    size_t max_idx = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
    return [NSString stringWithFormat: @"%zu (%f)", max_idx, outputs.at(max_idx)];}

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    self.indicatorView = [UIActivityIndicatorView new];
    [self.indicatorView setActivityIndicatorViewStyle:UIActivityIndicatorViewStyleGray];
    
    if (!predictor) {
        NSString *jsonPath = [[NSBundle mainBundle] pathForResource:SYMBOL_FILE ofType:nil];
        NSString *paramsPath = [[NSBundle mainBundle] pathForResource:PARAMS_FILE ofType:nil];
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
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(){
            [self prepareAndClassify:chosenImage.CGImage];
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
    
    if (!videoDevice) {
        videoDevice = [self selectCameraAt:AVCaptureDevicePositionBack];
    }
    
    if (!captureSession) {
        captureSession = [self createCaptureSessionFor:videoDevice];
    }
    
    self.exposureSlider.minimumValue = videoDevice.minExposureTargetBias;
    self.exposureSlider.maximumValue = videoDevice.maxExposureTargetBias;
    self.exposureSlider.value = videoDevice.exposureTargetBias;
    
    [captureSession startRunning];
}

- (IBAction)stopDetectionButtonTapped:(id)sender {
    [self.detectionButton setTitle:@"Start Detection" forState:UIControlStateNormal];
    [self.detectionButton addTarget:self
                             action:@selector(startDetectionButtonTapped:)
                   forControlEvents:UIControlEventTouchUpInside];
    
    [captureSession stopRunning];
}

- (AVCaptureSession *)createCaptureSessionFor:(AVCaptureDevice *)device
{
    AVCaptureSession *session = [[AVCaptureSession alloc] init];
    session.sessionPreset = AVCaptureSessionPresetHigh;
    
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

- (cv::Mat)cvMatGrayFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();//CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC1); // 8 bits per component, 1 channels
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cols,                       // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNone |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    return cvMat;
}

-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

- (UIImage *) doThresholding:(UIImage *) source
{
    cv::Mat gray = [self cvMatGrayFromUIImage: source];
    cv::Mat gray_inverted;
    cv::Mat thresholded;
    cv::bitwise_not(gray, gray_inverted);
    cv::threshold(gray_inverted, thresholded, 0, 0, cv::THRESH_TOZERO | cv::THRESH_OTSU);
    return [self UIImageFromCVMat: thresholded];
}

- (void) prepareAndClassify:(CGImageRef) cgImage
{
    float cropRectSize = 448;
    
    // box around detection area
    UIGraphicsBeginImageContextWithOptions(CGSizeMake(CGImageGetHeight(cgImage), CGImageGetWidth(cgImage)), NO, 1.0);
    CGContextRef context = UIGraphicsGetCurrentContext();
    UIGraphicsPushContext(context);
    [[UIImage imageWithCGImage: cgImage scale:0.0 orientation:UIImageOrientationRight] drawAtPoint:CGPointZero];
    UIGraphicsPopContext();
    double x = CGImageGetHeight(cgImage)/2.0 - cropRectSize/2.0;
    double y = CGImageGetWidth(cgImage)/2.0 - cropRectSize/2.0;
    CGRect cropRect = CGRectMake(x, y, cropRectSize, cropRectSize);
    CGContextSetLineWidth(context, 8);
    CGContextSetStrokeColorWithColor(context, [[ UIColor whiteColor ] CGColor]);
    CGContextStrokeRect(context, cropRect);
    UIImage* augmentedImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    // update ui
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(){
        dispatch_async(dispatch_get_main_queue(), ^(){
            [self.imageViewPhoto setImage: augmentedImage];
        });
    });
    
    // crop and threshold image
    UIImage *thresholded = [self doThresholding: [self cropCenterRect:augmentedImage toSize: cropRectSize] ];

    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(){
        if (![predictionRunningLock tryLock]) {
            return;
        }
        
        // classify pic
        NSString *classification = [self classifyNumber:thresholded];
        
        dispatch_async(dispatch_get_main_queue(), ^(){
            self.labelDescription.text = classification;
            [self.imageViewCrop setImage: thresholded];
        });
        
        [predictionRunningLock unlock];
    });
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection {
    
    CGImageRef image = [self imageFromSampleBuffer:sampleBuffer];
    [self prepareAndClassify: image];
    CGImageRelease(image);
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

- (IBAction)exposureSliderValueChanged:(id)sender {
    if (!videoDevice) {
        return;
    }
    
    [videoDevice lockForConfiguration:nil];
    
    if([videoDevice isExposureModeSupported:AVCaptureExposureModeCustom]){
        [videoDevice setExposureTargetBias:self.exposureSlider.value completionHandler: nil];    }
    
    [videoDevice unlockForConfiguration];
}


@end
