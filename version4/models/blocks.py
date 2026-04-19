################################################################################
# FOLDER: models
# FILE:   blocks.py
# PATH:   .\models\blocks.py
################################################################################

def add_conv_block(builder, C_in, C_out, kernel_size, stride, padding, parent_id):
    """
    Constructs a standard Conv -> BatchNorm -> ReLU sequence as graph nodes.
    """
    parents = [parent_id] if parent_id is not None else []
    
    conv = builder.add_node('conv', {
        'in_channels': C_in, 
        'out_channels': C_out, 
        'kernel_size': kernel_size, 
        'stride': stride, 
        'padding': padding
    }, parents)
    
    bn = builder.add_node('bn', {'num_features': C_out}, [conv])
    relu = builder.add_node('relu', {}, [bn])
    
    return relu


def add_res_block(builder, C_in, C_out, stride, parent_id):
    """
    Constructs a Residual Block with skip connections as graph nodes.
    """
    # 1. First Conv layer of the residual block
    conv1 = builder.add_node('conv', {
        'in_channels': C_in, 'out_channels': C_out, 'kernel_size': 3, 'stride': stride, 'padding': 1
    }, [parent_id])
    bn1 = builder.add_node('bn', {'num_features': C_out}, [conv1])
    relu1 = builder.add_node('relu', {}, [bn1])
    
    # 2. Second Conv layer
    conv2 = builder.add_node('conv', {
        'in_channels': C_out, 'out_channels': C_out, 'kernel_size': 3, 'stride': 1, 'padding': 1
    }, [relu1])
    bn2 = builder.add_node('bn', {'num_features': C_out}, [conv2])
    
    # 3. Shortcut Projection (if dimensions change)
    shortcut_id = parent_id
    if stride != 1 or C_in != C_out:
        s_conv = builder.add_node('conv', {
            'in_channels': C_in, 'out_channels': C_out, 'kernel_size': 1, 'stride': stride, 'padding': 0
        }, [parent_id])
        s_bn = builder.add_node('bn', {'num_features': C_out}, [s_conv])
        shortcut_id = s_bn
        
    # 4. Merge shortcut and path
    add_node = builder.add_node('add', {}, [bn2, shortcut_id])
    relu_out = builder.add_node('relu', {}, [add_node])
    
    return relu_out