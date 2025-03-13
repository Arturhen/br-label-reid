
import torch
from torchvision import transforms
import yaml
from PIL import Image
from typing import OrderedDict
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor:
    """
    Nova classe para extrair características de veículos usando o modelo MBR.
    """
    def __init__(self, config_path, weights_path, device=None):
        """
        Inicializa o extrator de características.
        
        Args:
            config_path: Caminho para o arquivo de configuração YAML
            weights_path: Caminho para os pesos do modelo
            device: Dispositivo para execução (cuda ou cpu)
        """
        # Configurar dispositivo
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Carregar configuração
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Configurar transformações
        self.transform = transforms.Compose([
            transforms.Resize((self.config.get('y_length', 256), self.config.get('x_length', 256))),
            transforms.ToTensor(),
            transforms.Normalize(
                self.config.get('n_mean', [0.485, 0.456, 0.406]), 
                self.config.get('n_std', [0.229, 0.224, 0.225])
            ),
        ])
        
        # Carregar modelo
        self.model = get_model(self.config, self.device)
        
        # Carregar pesos
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        except RuntimeError as e:
            print(f"Erro ao carregar pesos diretamente: {e}")
            print("Tentando remover prefixo 'module.'...")
            weights = torch.load(weights_path, map_location=self.device)
            if isinstance(weights, dict) and 'state_dict' in weights:
                weights = weights['state_dict']
            weights = OrderedDict((k.replace("module.", ""), v) for k, v in weights.items())
            self.model.load_state_dict(weights)
            
        self.model.eval()
        print(f"Modelo carregado com sucesso de: {weights_path}")
    
    def __call__(self, image_path):
        """
        Extrai características de uma imagem.
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Vetor de características normalizado
        """
        # Carregar e preprocessar imagem
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)


        # Definir IDs padrão de câmera e viewpoint
        cam_id = torch.tensor([0]).to(self.device) #AQUI DEVERIA SER MELHOR USADO
        view_id = torch.tensor([0]).to(self.device) #AQUI DEVERIA SER MELHOR USADO 
        # Extrair características
        with torch.no_grad():
            _, _, ffs, _ = self.model(image, cam_id, view_id)
            
            # Normalizar e concatenar vetores de características
            end_vec = []
            for item in ffs:
                end_vec.append(F.normalize(item))
            features = torch.cat(end_vec, 1)
        
        return features.cpu()




def get_model(data, device):

    ### 2B hybrid No LBS   
    if 'Hybrid_2B' == data['model_arch']:
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "BoT"], n_groups=0, losses="Classical", LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 2B R50 No LBS
    if 'R50_2B' == data['model_arch']:
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 2B R50 LBS
    if data['model_arch'] == 'MBR_R50_2B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50"], losses="LBS", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### Baseline with BoT
    if data['model_arch'] == 'BoT_baseline':
        model = MBR_model(class_num=data['n_classes'], n_branches=["BoT"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 2B BoT LBS
    if data['model_arch'] == 'MBR_BOT_2B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["BoT", "BoT"], losses="LBS", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### MBR-4B (4B hybrid LBS)
    if data['model_arch'] == 'MBR_4B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50", "BoT", "BoT"], losses="LBS", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
    
    ### 4B hybdrid No LBS
    if data['model_arch'] == 'Hybrid_4B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50", "BoT", "BoT"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 4B R50 No LBS
    if data['model_arch'] == 'R50_4B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50", "R50", "R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])    

    if data['model_arch'] == 'MBR_R50_4B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50", "R50", "R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 4G hybryd with LBS     MBR-4G
    if data['model_arch'] =='MBR_4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 4G hybrid No LBS
    if data['model_arch'] =='Hybrid_4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    if data['model_arch'] =='MBR_2x2G':    
        model = MBR_model(class_num=data['n_classes'], n_branches=['2x'], losses="LBS", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], x2g=True, group_conv_mhsa_2=True) 

    if data['model_arch'] =='MBR_R50_2x2G':  
        model = MBR_model(class_num=data['n_classes'], n_branches=['2x'], losses="LBS", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], x2g=True)  

    ### 2G BoT LBS
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], end_bot_g=True)

    ### 2G R50 LBS
    if data['model_arch'] =='MBR_R50_2G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 2G Hybrid No LBS
    if data['model_arch'] =='Hybrid_2G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="Classical", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], group_conv_mhsa_2=True)

    ### 2G R50 No LBS
    if data['model_arch'] =='R50_2G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="Classical", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 4G R50 No LBS
    if data['model_arch'] =='R50_4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="Classical", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    ### 4G only R50 with LBS
    if data['model_arch'] =='MBR_R50_4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], group_conv_mhsa_2=True)
    
    if data['model_arch'] =='MBR_R50_2x4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=["2x"], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], x4g=True)

    if data['model_arch'] =='MBR_2x4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=["2x"], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], x4g=True, group_conv_mhsa=True)

    if data['model_arch'] == 'Baseline':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    return model.to(device)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class MBR_model(nn.Module):         
    def __init__(self, class_num, n_branches, n_groups, losses="LBS", backbone="ibn", droprate=0, linear_num=False, return_f = True, circle_softmax=False, pretrain_ongroups=True, end_bot_g=False, group_conv_mhsa=False, group_conv_mhsa_2=False, x2g=False, x4g=False, LAI=False, n_cams=0, n_views=0):
        super(MBR_model, self).__init__()  

        self.modelup2L3 = base_branches(backbone=backbone)
        self.modelL4 = multi_branches(n_branches=n_branches, n_groups=n_groups, pretrain_ongroups=pretrain_ongroups, end_bot_g=end_bot_g, group_conv_mhsa=group_conv_mhsa, group_conv_mhsa_2=group_conv_mhsa_2, x2g=x2g, x4g=x4g)
        self.finalblock = FinalLayer(class_num=class_num, n_branches=n_branches, n_groups=n_groups, losses=losses, droprate=droprate, linear_num=linear_num, return_f=return_f, circle_softmax=circle_softmax, LAI=LAI, n_cams=n_cams, n_views=n_views, x2g=x2g, x4g=x4g)
        

    def forward(self, x,cam, view):
        mix = self.modelup2L3(x)
        output = self.modelL4(mix)
        preds, embs, ffs = self.finalblock(output, cam, view)

        return preds, embs, ffs, output

class base_branches(nn.Module):
    def __init__(self, backbone="ibn", stride=1):
        super(base_branches, self).__init__()
        if backbone == 'r50':
            raise ValueError("Please use IBN models")
            # model_ft = models.resnet50()
        elif backbone == '101ibn':
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=True)# 'resnet50_ibn_a'
        elif backbone == '34ibn':
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet34_ibn_a', pretrained=True)# 'resnet50_ibn_a'
        else:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
            
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            if backbone == "34ibn":
                model_ft.layer4[0].conv1.stride = (1,1)
            else:
                model_ft.layer4[0].conv2.stride = (1,1)

        self.model = torch.nn.Sequential(*(list(model_ft.children())[:-3])) 

    def forward(self, x):
        x = self.model(x)
        return x
    
class multi_branches(nn.Module):
    def __init__(self, n_branches, n_groups, pretrain_ongroups=True, end_bot_g=False, group_conv_mhsa=False, group_conv_mhsa_2=False, x2g = False, x4g=False):
        super(multi_branches, self).__init__()

        model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        model_ft= model_ft.layer4
        self.x2g = x2g
        self.x4g = x4g
        if n_groups > 0:
            convlist = [k.split('.') for k, m in model_ft.named_modules(remove_duplicate=False) if isinstance(m, nn.Conv2d)]
            for item in convlist:
                if item[1] == "downsample":
                    m = model_ft[int(item[0])].get_submodule(item[1])[0]
                else:
                    m = model_ft[int(item[0])].get_submodule(item[1]) #'.'.join(
                weight = m.weight[:int(m.weight.size(0)), :int(m.weight.size(1)/n_groups), :,:]
                if end_bot_g and item[1]=="conv2":
                    raise ValueError("Please use 4G or 2G models")
                    # setattr(model_ft[int(item[0])], item[1], MHSA_2G(int(512), int(512)))
                elif group_conv_mhsa and item[1]=="conv2":
                    raise ValueError("Please use 4G or 2G models")
                    # setattr(model_ft[int(item[0])], item[1], Conv_MHSA_4G(int(512), int(512)))
                elif group_conv_mhsa_2 and item[1]=="conv2":
                    raise ValueError("Please use 4G or 2G models")
                    # setattr(model_ft[int(item[0])], item[1], Conv_MHSA_2G(int(512), int(512)))
                else:
                    if item[1] == "downsample":
                        getattr(model_ft[int(item[0])], item[1])[0] = nn.Conv2d(int(m.weight.size(1)), int(m.weight.size(0)), kernel_size=1, stride=1, groups=n_groups, bias=False).apply(weights_init_kaiming)
                        if pretrain_ongroups:
                            getattr(model_ft[int(item[0])], item[1])[0].weight.data = weight
                    elif item[1] == "conv2":
                        setattr(model_ft[int(item[0])], item[1], nn.Conv2d(int(m.weight.size(1)), int(m.weight.size(0)), kernel_size=3, stride=1, padding=(1,1), groups=n_groups, bias=False).apply(weights_init_kaiming))
                        if pretrain_ongroups:
                            setattr(model_ft[int(item[0])].get_submodule(item[1]).weight, "data", weight)                        
                    else:
                        setattr(model_ft[int(item[0])], item[1], nn.Conv2d(int(m.weight.size(1)), int(m.weight.size(0)), kernel_size=1, stride=1, groups=n_groups, bias=False).apply(weights_init_kaiming))
                        if pretrain_ongroups:
                            setattr(model_ft[int(item[0])].get_submodule(item[1]).weight, "data", weight)
        self.model = nn.ModuleList()

        if len(n_branches) > 0:
            raise ValueError("Please use 4G or 2G models")
            # if n_branches[0] == "2x":
            #     self.model.append(model_ft)
            #     self.model.append(copy.deepcopy(model_ft))
            # else:
            #     for item in n_branches:
            #         if item =="R50":
            #             self.model.append(copy.deepcopy(model_ft))
            #         elif item == "BoT":
            #             layer_0 = Bottleneck_Transformer(1024, 512, resolution=[16, 16], use_mlp = False)
            #             layer_1 = Bottleneck_Transformer(2048, 512, resolution=[16, 16], use_mlp = False)
            #             layer_2 = Bottleneck_Transformer(2048, 512, resolution=[16, 16], use_mlp = False)
            #             self.model.append(nn.Sequential(layer_0, layer_1, layer_2))
            #         else:
            #             print("No valid architecture selected for branching by expansion!")
        else:
            self.model.append(model_ft)


    def forward(self, x):
        output = []
        for cnt, branch in enumerate(self.model):
            if self.x2g and cnt>0:
                aux = torch.cat((x[:,int(x.shape[1]/2):,:,:], x[:,:int(x.shape[1]/2),:,:]), dim=1)
                output.append(branch(aux))
            elif self.x4g and cnt>0:
                aux = torch.cat((x[:,int(x.shape[1]/4):int(x.shape[1]/4*2),:,:], x[:, :int(x.shape[1]/4),:,:], x[:, int(x.shape[1]/4*3):,:,:], x[:, int(x.shape[1]/4*2):int(x.shape[1]/4*3),:,:]), dim=1)
                output.append(branch(aux))
            else:
                output.append(branch(x))
       
        return output

class FinalLayer(nn.Module):
    def __init__(self, class_num, n_branches, n_groups, losses="LBS", droprate=0, linear_num=False, return_f = True, circle_softmax=False, n_cams=0, n_views=0, LAI=False, x2g=False,x4g=False):
        super(FinalLayer, self).__init__()    
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.finalblocks = nn.ModuleList()
        self.withLAI = LAI
        if n_groups > 0:
            self.n_groups = n_groups
            for i in range(n_groups*(len(n_branches)+1)):
                if losses == "LBS":
                    if i%2==0:
                        self.finalblocks.append(ClassBlock(int(2048/n_groups), class_num, droprate, linear=linear_num, return_f = return_f, circle=circle_softmax))
                    else:
                        bn= nn.BatchNorm1d(int(2048/n_groups))
                        bn.bias.requires_grad_(False)  
                        bn.apply(weights_init_kaiming)
                        self.finalblocks.append(bn)
                else:
                    self.finalblocks.append(ClassBlock(int(2048/n_groups), class_num, droprate, linear=linear_num, return_f = return_f, circle=circle_softmax))
        else:
            self.n_groups = 1
            for i in range(len(n_branches)):
                if losses == "LBS":
                    if i%2==0:
                        self.finalblocks.append(ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = return_f, circle=circle_softmax))
                    else:
                        bn= nn.BatchNorm1d(int(2048))
                        bn.bias.requires_grad_(False)  
                        bn.apply(weights_init_kaiming)
                        self.finalblocks.append(bn)
                else:
                    self.finalblocks.append(ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = return_f, circle=circle_softmax))

        if losses == "LBS":
            self.LBS = True
        else:
            self.LBS = False

        if self.withLAI:
            # self.LAI = []
            self.n_cams = n_cams
            self.n_views = n_views
            if n_groups>0 and len(n_branches)==0:
                n_branches = ["groups"]
            if n_cams>0 and n_views>0:
                if x2g or x4g:
                    self.LAI = nn.Parameter(torch.zeros(2, n_cams * n_views, 2048))
                else:
                    self.LAI = nn.Parameter(torch.zeros(len(n_branches), n_cams * n_views, 2048))
            elif n_cams>0:
                if x2g or x4g:
                    self.LAI = nn.Parameter(torch.zeros(2, n_cams, 2048))
                else:
                    self.LAI = nn.Parameter(torch.zeros(len(n_branches), n_cams, 2048))
            elif n_views>0:
                if x2g or x4g:
                    self.LAI = nn.Parameter(torch.zeros(2, n_views, 2048))
                else:
                    self.LAI = nn.Parameter(torch.zeros(len(n_branches), n_views, 2048))
            else: self.withLAI = False

    def forward(self, x, cam, view):
        # if len(x) != len(self.finalblocks):
        #     print("Something is wrong")
        embs = []
        ffs = []
        preds = []
        for i in range(len(x)):
            emb = self.avg_pool(x[i]).squeeze(dim=-1).squeeze(dim=-1)
            if self.withLAI:
                if self.n_cams > 0 and self.n_views >0:
                    emb = emb + self.LAI[i, cam * self.n_views + view, :]
                elif self.n_cams >0:
                    emb = emb + self.LAI[i, cam, :]
                else:
                    emb = emb + self.LAI[i, view, :]
            for j in range(self.n_groups):
                aux_emb = emb[:,int(2048/self.n_groups*j):int(2048/self.n_groups*(j+1))]
                if self.LBS:
                    if (i+j)%2==0:
                        pred, ff = self.finalblocks[i+j](aux_emb)
                        ffs.append(ff)
                        preds.append(pred)
                    else:
                        ff = self.finalblocks[i+j](aux_emb)
                        embs.append(aux_emb)
                        ffs.append(ff)
                else:
                    aux_emb = emb[:,int(2048/self.n_groups*j):int(2048/self.n_groups*(j+1))]
                    pred, ff = self.finalblocks[i+j](aux_emb)
                    embs.append(aux_emb)
                    ffs.append(ff)
                    preds.append(pred)
                    
        return preds, embs, ffs

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.0, relu=False, bnorm=True, linear=False, return_f = True, circle=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        self.circle = circle
        add_block = []
        if linear: ####MLP to reduce
            final_dim = linear
            add_block += [nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, final_dim)]
        else:
            final_dim = input_dim
        if bnorm:
            tmp_block = nn.BatchNorm1d(final_dim)
            tmp_block.bias.requires_grad_(False) 
            add_block += [tmp_block]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(final_dim, class_num, bias=False)] # 
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        if x.dim()==4:
            x = x.squeeze().squeeze()
        if x.dim()==1:
            x = x.unsqueeze(0)
        x = self.add_block(x)
        if self.return_f:
            f = x
            if self.circle:
                x = F.normalize(x)
                self.classifier[0].weight.data = F.normalize(self.classifier[0].weight, dim=1)
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x

