import os
import torch
from torch import nn
from torch.autograd import Variable

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def conv_block(in_c, out_c, name, max_pooling, bn=True, relu=True, size=4, pad=1, dropout=0.):
    block = nn.Sequential()
    block.add_module('%s_tconv1' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=1, padding=pad))
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    block.add_module('%s_tconv2' % name, nn.Conv2d(out_c, out_c, kernel_size=size, stride=1, padding=pad))
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))

    if max_pooling:
        block.add_module('%s_maxpooling' % name, nn.MaxPool2d(kernel_size=size, stride=2, padding=pad, dilation = 1))

    return block


def upsampling_block(in_c, out_c, name, relu=True, size=4, pad=1):
    block = nn.Sequential()
    block.add_module('%s_tconv1' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=1, padding=pad))
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    block.add_module('%s_tconv2' % name, nn.Conv2d(out_c, out_c, kernel_size=size, stride=1, padding=pad))
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    return block


class TurbNetG_vel(nn.Module):
    def __init__(self, channelExponent=3, dropout=0.):
        super(TurbNetG_vel, self).__init__()
        channels = int(2 ** channelExponent + 0.5)
        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(2, channels, 3, 1, 1))

        self.layer1b = conv_block(channels, channels*2, 'layer1b', max_pooling=True, bn=True, relu=True, size=3, pad=1, dropout=dropout)
        self.layer2 = conv_block(channels*2, channels*4, 'layer2', bn=True, relu=True, size=3, pad=1, dropout=dropout, max_pooling=True)
        self.layer3 = conv_block(channels*4, channels*8, 'layer3', bn=True, relu=True, size=3, pad=1, dropout=dropout, max_pooling=True)
        self.layer4 = conv_block(channels*8, channels*16, 'layer4', bn=True, relu=True, size=3, pad=1, dropout=dropout, max_pooling=True)
        self.layer5 = conv_block(channels*16, channels*32, 'layer5', bn=True, relu=True, size=3, pad=1, dropout=dropout, max_pooling=True)
        self.layer6 = conv_block(channels*32, channels*64, 'layer6', bn=True, relu=True, size=3, pad=1, dropout=dropout, max_pooling=True)

        # note, kernel size is internally reduced by one now
        self.dlayer6 = nn.Sequential()
        self.dlayer6.add_module('%dlayer6_transpose_conv', nn.ConvTranspose2d(channels*64, channels*32, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.dlayer6b = upsampling_block(channels*64, channels*32, 'dlayer6', relu=True, size=3, pad=1)
        self.dlayer5 = nn.Sequential()
        self.dlayer5.add_module('%dlayer5_transpose_conv', nn.ConvTranspose2d(channels*32, channels*16, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.dlayer5b = upsampling_block(channels * 32, channels * 16, 'dlayer5', relu=True, size=3, pad=1)
        self.dlayer4 = nn.Sequential()
        self.dlayer4.add_module('%dlayer4_transpose_conv', nn.ConvTranspose2d(channels*16, channels*8, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.dlayer4b = upsampling_block(channels * 16, channels * 8, 'dlayer4', relu=True, size=3, pad=1)
        self.dlayer3 = nn.Sequential()
        self.dlayer3.add_module('%dlayer3_transpose_conv', nn.ConvTranspose2d(channels*8, channels*4, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.dlayer3b = upsampling_block(channels * 8, channels * 4, 'dlayer3', relu=True, size=3, pad=1)
        self.dlayer2 = nn.Sequential()
        self.dlayer2.add_module('%dlayer2_transpose_conv', nn.ConvTranspose2d(channels*4, channels*2, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.dlayer2b = upsampling_block(channels * 4, channels * 2, 'dlayer2', relu=True, size=3, pad=1)
        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_conv', nn.ConvTranspose2d(channels*2, 2, kernel_size=3, stride=2, padding=1, output_padding=1))

    def forward(self, x):
        out1 = self.layer1(x)
        out1b = self.layer1b(out1)
        out2 = self.layer2(out1b)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout6b = self.dlayer6b(dout6_out5)
        dout5 = self.dlayer5(dout6b)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout5b = self.dlayer5b(dout5_out4)
        dout4 = self.dlayer4(dout5b)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout4b = self.dlayer4b(dout4_out3)
        dout3 = self.dlayer3(dout4b)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout3b = self.dlayer3b(dout3_out2)
        dout2 = self.dlayer2(dout3b)
        dout2_out1b = torch.cat([dout2, out1b], 1)
        dout2b = self.dlayer2b(dout2_out1b)
        dout1 = self.dlayer1(dout2b)

        return dout1


class TurbNetG_pressure(nn.Module):
    def __init__(self, channelExponent=3, dropout=0.):
        super(TurbNetG_pressure, self).__init__()
        channels = int(2 ** channelExponent + 0.5)
        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(2, channels, 3, 1, 1))

        self.layer1b = conv_block(channels, channels*2, 'layer1b', max_pooling=True, bn=True, relu=True, size=3, pad=1, dropout=dropout)
        self.layer2 = conv_block(channels*2, channels*4, 'layer2', bn=True, relu=True, size=3, pad=1, dropout=dropout, max_pooling=True)
        self.layer3 = conv_block(channels*4, channels*8, 'layer3', bn=True, relu=True, size=3, pad=1, dropout=dropout, max_pooling=True)
        self.layer4 = conv_block(channels*8, channels*16, 'layer4', bn=True, relu=True, size=3, pad=1, dropout=dropout, max_pooling=True)
        self.layer5 = conv_block(channels*16, channels*32, 'layer5', bn=True, relu=True, size=3, pad=1, dropout=dropout, max_pooling=True)
        self.layer6 = conv_block(channels*32, channels*64, 'layer6', bn=True, relu=True, size=3, pad=1, dropout=dropout, max_pooling=True)

        # note, kernel size is internally reduced by one now
        self.dlayer6 = nn.Sequential()
        self.dlayer6.add_module('%dlayer6_transpose_conv', nn.ConvTranspose2d(channels*64, channels*32, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.dlayer6b = upsampling_block(channels*64, channels*32, 'dlayer6', relu=True, size=3, pad=1)
        self.dlayer5 = nn.Sequential()
        self.dlayer5.add_module('%dlayer5_transpose_conv', nn.ConvTranspose2d(channels*32, channels*16, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.dlayer5b = upsampling_block(channels * 32, channels * 16, 'dlayer5', relu=True, size=3, pad=1)
        self.dlayer4 = nn.Sequential()
        self.dlayer4.add_module('%dlayer4_transpose_conv', nn.ConvTranspose2d(channels*16, channels*8, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.dlayer4b = upsampling_block(channels * 16, channels * 8, 'dlayer4', relu=True, size=3, pad=1)
        self.dlayer3 = nn.Sequential()
        self.dlayer3.add_module('%dlayer3_transpose_conv', nn.ConvTranspose2d(channels*8, channels*4, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.dlayer3b = upsampling_block(channels * 8, channels * 4, 'dlayer3', relu=True, size=3, pad=1)
        self.dlayer2 = nn.Sequential()
        self.dlayer2.add_module('%dlayer2_transpose_conv', nn.ConvTranspose2d(channels*4, channels*2, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.dlayer2b = upsampling_block(channels * 4, channels * 2, 'dlayer2', relu=True, size=3, pad=1)
        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_conv', nn.ConvTranspose2d(channels*2, 1, kernel_size=3, stride=2, padding=1, output_padding=1))

    def forward(self, x):
        out1 = self.layer1(x)
        out1b = self.layer1b(out1)
        out2 = self.layer2(out1b)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout6b = self.dlayer6b(dout6_out5)
        dout5 = self.dlayer5(dout6b)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout5b = self.dlayer5b(dout5_out4)
        dout4 = self.dlayer4(dout5b)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout4b = self.dlayer4b(dout4_out3)
        dout3 = self.dlayer3(dout4b)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout3b = self.dlayer3b(dout3_out2)
        dout2 = self.dlayer2(dout3b)
        dout2_out1b = torch.cat([dout2, out1b], 1)
        dout2b = self.dlayer2b(dout2_out1b)
        dout1 = self.dlayer1(dout2b)

        return dout1


def predict_ver_pressure(inputDF):
    net = TurbNetG_pressure()
    net.load_state_dict(torch.load("./saved_models/model_pressure_1000", map_location=torch.device('cpu')))
    net.eval()

    # Generate prediction
    inputs = torch.FloatTensor(1, 2, 128, 128)
    inputs = Variable(inputs)
    inputs.data.copy_(torch.from_numpy(inputDF))

    prediction = net(inputs)

    return prediction


def predict_hor_pressure(inputDF):
    net = TurbNetG_pressure()
    net.load_state_dict(torch.load("./saved_models/trained_models_model_pressure_500_hor", map_location=torch.device('cpu')))
    net.eval()

    # Generate prediction
    inputs = torch.FloatTensor(1, 2, 128, 128)
    inputs = Variable(inputs)
    inputs.data.copy_(torch.from_numpy(inputDF))

    prediction = net(inputs)

    return prediction


def predict_hor_vel(inputDF):
    net = TurbNetG_vel()
    net.load_state_dict(torch.load("./saved_models/trained_models_model_velocity_alle", map_location=torch.device('cpu')))
    net.eval()

    # Generate prediction
    inputs = torch.FloatTensor(1, 2, 128, 128)
    inputs = Variable(inputs)
    inputs.data.copy_(torch.from_numpy(inputDF))

    #streamlit.write(inputs.data.shape, inputs.data[0,0,:,:])
    #streamlit.write(inputs.data.shape, inputs.data[0,1,:,:])
    #streamlit.write(inputs.data.shape, inputs.data[0,2,:,:])
    #streamlit.write(inputs.data.shape, inputs.data[0,3,:,:])

    prediction = net(inputs)

    return prediction


def predict_ver_vel(inputDF):
    net = TurbNetG_vel()
    net.load_state_dict(torch.load("./saved_models/model_velocity_alle", map_location=torch.device('cpu')))
    net.eval()

    # Generate prediction
    inputs = torch.FloatTensor(1, 2, 128, 128)
    inputs = Variable(inputs)
    inputs.data.copy_(torch.from_numpy(inputDF))

    prediction = net(inputs)

    return prediction


