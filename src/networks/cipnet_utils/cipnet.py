import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet50_features_inat, resnet101_features, resnet152_features
from networks.convnext_features import convnext_tiny_26_features, convnext_tiny_13_features 
from networks.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from torch import Tensor
        
class PIPNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_prototypes: int,
                 feature_net: nn.Module,
                 add_on_layers: nn.Module,
                 pool_layer: nn.Module,
                 classification_layer: nn.Module
                 ):
        super().__init__()
        assert num_classes > 0
        # self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier
        
        # self.pipnet_head = PIPNet_head(self._num_classes,
        #                                self._num_prototypes,
        #                                self._net,
        #                                args,
        #                                self._add_on,
        #                                self._pool,
        #                                self._classification)

    def forward(self, xs,  inference=False):
        # DIMENSIONI XS IN FORWARD PIPNET: torch.Size([64, 3, 224, 224])
        features = self._net(xs) # [64, 3, 224, 224]
        # DIMENSIONI FEATURES IN FORWARD PIPNET: torch.Size([64, 512, 28, 28]) se resnet / torch.Size([64, 768, 26, 26]) se convnext_tiny_26
        proto_features = self._add_on(features)
        # DIMENSIONI PROTOFEATURES IN FORWARD PIPNET: torch.Size([64, 512, 28, 28]) se resnet / torch.Size([64, 768, 26, 26]) se convnext_tiny_26
        pooled = self._pool(proto_features)
        # DIMENSIONI pooled IN FORWARD PIPNET: torch.Size([64, 512]) se resnet / torch.Size([64, 768])se convnext_tiny_26
        if inference:
            clamped_pooled = torch.where(pooled < 0.1, 0., pooled)  #during inference, ignore all prototypes that have 0.1 similarity or lower
            out = self._classification(clamped_pooled) #shape (bs*2, num_classes)
            return proto_features, clamped_pooled, out
        else:
            out = self._classification(pooled) #shape (bs*2, num_classes) 
            # DIMENSIONI out IN FORWARD PIPNET: torch.Size([64, 200])
            return proto_features, pooled, out

#Complete CIPNet model used in the paper
class CIPNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_prototypes: int,
                 feature_net: nn.Module,
                 add_on_layers: nn.Module,
                 pool_layer: nn.Module,
                 classification_layer: nn.Module
                 ):
        # super().__init__(num_classes, num_prototypes, feature_net, args, add_on_layers, pool_layer, classification_layer)
        super().__init__()
        assert num_classes > 0
        # self._num_features = args.num_features
        self._num_classes = num_classes
        # print("num_classes per task", num_classes)
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        self._classification = nn.ModuleList([classification_layer])
        # self._multiplier = [classification_layer.normalization_multiplier]
        self.tau = nn.Parameter(torch.tensor(16.0), requires_grad = True) #Then we freeze it or unfreeze it ###

    def add_head(self, bias=False):
        self._classification.append(CosineLinear(self._num_prototypes, self._num_classes).to(self._classification[-1].weight.device))
        if bias:
            torch.nn.init.constant_(self._classification[-1].bias, val=0.)
        print("CIPNET CLS HEAD ADDED AND INITIALIZED")

    def forward(self, xs,  inference=False, task=None):
        # DIMENSIONI XS IN FORWARD CIPNET: torch.Size([64, 3, 224, 224])
        features = self._net(xs) # DOVE NET SAREBBE LA RETE CONVOLUZIONALE # [64, 3, 224, 224]
        # DIMENSIONI FEATURES IN FORWARD CIPNET: torch.Size([64, 512, 28, 28]) se resnet / torch.Size([64, 768, 26, 26]) se convnext_tiny_26
        proto_features = self._add_on(features)
        # DIMENSIONI PROTOFEATURES IN FORWARD CIPNET: torch.Size([64, 512, 28, 28]) se resnet / torch.Size([64, 768, 26, 26]) se convnext_tiny_26
        pooled = self._pool(proto_features)
        # DIMENSIONI pooled IN FORWARD CIPNET: torch.Size([64, 512]) se resnet / torch.Size([64, 768])se convnext_tiny_26
        if inference:
            pooled = torch.where(pooled < 0.1, 0., pooled)  #during inference, ignore all prototypes that have 0.1 similarity or lower
        if task is not None:
            out = self._classification[task](pooled, self.tau) ###
        else:
            out = []
            for i in range(len(self._classification)):
                out.append(self._classification[i](pooled, self.tau)) ###
            out = torch.cat(out, dim=1)
        return proto_features, pooled, out


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 #'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'convnext_tiny_26': convnext_tiny_26_features,
                                 #'convnext_tiny_13': convnext_tiny_13_features
                                }




# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,),requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        # return F.linear(input,torch.relu(self.weight), self.bias)
        return F.linear(input,torch.exp(self.weight), self.bias)
    
#Classification head used in the paper
class CosineLinear(nn.Module):
    """
    Cosine-similarity classifier with a shared temperature τ.
    No bias term (bias would break the scale invariance).
    """
    def __init__(self, in_features: int, out_features: int, tau_init=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.normal_(self.weight, mean=1.0,std=0.1)

    def forward(self, x, tau=torch.tensor(16.0)):
        w = F.softplus(self.weight) + 1e-6
        # L2-normalise features and weights
        x_norm  = F.normalize(x,  dim=1)
        w_norm  = F.normalize(w, dim=1)
        return tau * F.linear(x_norm, w_norm)

def get_network(num_classes: int, network, pretrained, bias=False): 
    # features = base_architecture_to_features[args.network](pretrained=not args.disable_pretrained)
    features = base_architecture_to_features[network](pretrained=pretrained)
    features_name = str(features).upper()
    if 'next' in network:
        features_name = str(network).upper()
    if features_name.startswith('RES') or features_name.startswith('CONVNEXT'):
        print("Using", features_name, "as base architecture", flush=True)
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
    else:
        raise Exception('other base architecture NOT implemented')
    
    num_prototypes = first_add_on_layer_in_channels
    print("Number of prototypes: ", num_prototypes, flush=True)
    add_on_layers = nn.Sequential(
        nn.Softmax(dim=1), #softmax over every prototype for each patch, 
                               #such that for every location in image, sum over prototypes is 1                
    )
    
    pool_layer = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=(1,1)), #outputs (bs, ps,1,1)
                nn.Flatten() #outputs (bs, ps)
                ) 
    print("NUM PROTOTYPES",num_prototypes)
    
    classification_layer = CosineLinear(num_prototypes, num_classes)
        
    return features, add_on_layers, pool_layer, classification_layer, num_prototypes


def construct_CIPNet(num_classes, network, pretrained=True, bias=False):
        
    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(num_classes, network, pretrained, bias)
    return CIPNet(num_classes=num_classes,
                    num_prototypes=num_prototypes,
                    feature_net = feature_net,
                    add_on_layers = add_on_layers,
                    pool_layer = pool_layer,
                    classification_layer = classification_layer
                 )   
