################################################################################
# FOLDER: models
# FILE:   cells.py
# PATH:   .\models\cells.py
################################################################################

from models.blocks import add_conv_block, add_res_block

def add_cell(builder, C_in, C_out, stride, parent_id):
    """
    Constructs a Normal Cell. Adapts dimensions first if necessary.
    """
    if C_in != C_out or stride != 1:
        # 1x1 Convolution to adjust channels/spatial dimensions
        parent_id = add_conv_block(builder, C_in, C_out, kernel_size=1, stride=stride, padding=0, parent_id=parent_id)
        
    # Standard residual feature extraction
    return add_res_block(builder, C_out, C_out, stride=1, parent_id=parent_id)


def add_reduction_cell(builder, C_in, C_out, parent_id):
    """
    Constructs a Reduction Cell to halve spatial dimensions and double channels.
    """
    return add_conv_block(builder, C_in, C_out, kernel_size=3, stride=2, padding=1, parent_id=parent_id)