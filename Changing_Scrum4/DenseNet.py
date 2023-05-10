from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Rescaling, Input

def build_densenet_model(input_shape=(256,256,3),classes=1,load_weights =None,type='multi_label'):
    """use for building ResNet101 architecture which has Rescaling layer after input layer and change output layer by type of prediction
        type mean type of label, 'multi_label' or 'multi_class'
    """
    densenet = DenseNet121(include_top=False,weights=load_weights,input_shape=input_shape,classes=classes,pooling='avg')
    
    if type == 'multi_label':
        out_activation = 'sigmoid'
        out = tf.keras.layers.Dense(classes,activation=out_activation)(densenet.output)
        model = tf.keras.models.Model(inputs=densenet.input, outputs=out)
    else:
        out_activation = 'softmax'
        out = tf.keras.layers.Dense(classes,activation=out_activation)(densenet.output)
        base_model = tf.keras.models.Model(inputs=densenet.input, outputs=out)
        input = Input(input_shape)
        scaled_inp = Rescaling(1./255)(input)
        output = base_model(scaled_inp)
        model = tf.keras.models.Model(inputs= input, outputs=output)
        
    
    model.summary()
    
    return model