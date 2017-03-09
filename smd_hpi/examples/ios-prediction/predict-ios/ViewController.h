//
//  ViewController.h
//  WhatIsThis
//
//  Created by Haoxiang Li on 1/23/16.
//  Copyright © 2016 Haoxiang Li. All rights reserved.
//

#import <UIKit/UIKit.h>
#import "c_predict_api.h"
#import <vector>
#import <AVFoundation/AVCaptureOutput.h> // Allows us to use AVCaptureVideoDataOutputSampleBufferDelegate

#define kDefaultWidth 224
#define kDefaultHeight 224
#define kDefaultChannels 3
#define kDefaultImageSize (kDefaultWidth*kDefaultHeight*kDefaultChannels)

@interface ViewController : UIViewController <AVCaptureVideoDataOutputSampleBufferDelegate>{
    
    PredictorHandle predictor;
    
    NSString *model_symbol;
    NSData *model_params;
    NSMutableArray *model_synset;
    float model_mean[kDefaultImageSize];
    AVCaptureSession *captureSession;
}

@property (weak, nonatomic) IBOutlet UILabel *labelDescription;
@property (weak, nonatomic) IBOutlet UIImageView *imageViewPhoto;
@property (weak, nonatomic) IBOutlet UIButton *detectionButton;

- (IBAction)startDetectionButtonTapped:(id)sender;
- (IBAction)stopDetectionButtonTapped:(id)sender;
- (AVCaptureSession *)createCaptureSession;

@end

