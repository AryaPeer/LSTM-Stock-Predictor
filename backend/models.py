# Models

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, GRU,
    Attention, Dropout, Dense, Flatten, BatchNormalization,
    MultiHeadAttention, LayerNormalization, Add, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union
from abc import ABC, abstractmethod
import logging
import hashlib
import json
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    CNN_BILSTM_ATTENTION = "cnn_bilstm_attention"
    TRANSFORMER = "transformer"
    GRU_ATTENTION = "gru_attention"
    WAVENET = "wavenet"
    ENSEMBLE = "ensemble"


class ScalerType(Enum):
    MINMAX = "minmax"
    STANDARD = "standard"
    ROBUST = "robust"


@dataclass
class ModelConfig:
    model_type: ModelType = ModelType.CNN_BILSTM_ATTENTION
    time_step: int = 60
    future_steps: int = 10

    # Architecture parameters
    cnn_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    lstm_units: List[int] = field(default_factory=lambda: [256, 128, 64])
    gru_units: List[int] = field(default_factory=lambda: [128, 64])
    dropout_rates: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4])
    attention_heads: int = 8

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 200
    validation_split: float = 0.2
    patience: int = 15
    min_delta: float = 0.0001

    # Regularization
    l1_reg: float = 0.0001
    l2_reg: float = 0.001

    # Advanced features
    use_batch_norm: bool = True
    use_residual: bool = True
    use_ensemble: bool = False
    ensemble_models: int = 3

    # Scaling
    scaler_type: ScalerType = ScalerType.ROBUST

    # Random seed
    random_seed: int = 42

    def to_dict(self) -> Dict:
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
        }

    def get_hash(self) -> str:
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


class BaseModel(ABC):

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.history = None
        self._set_seeds()

    def _set_seeds(self):
        np.random.seed(self.config.random_seed)
        tf.random.set_seed(self.config.random_seed)

    @abstractmethod
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        pass

    def compile_model(self):
        if self.model is None:
            raise ValueError("Model not built yet")

        optimizer = Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=1.0
        )

        self.model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse']
        )

    def get_scaler(self):
        if self.config.scaler_type == ScalerType.MINMAX:
            return MinMaxScaler()
        elif self.config.scaler_type == ScalerType.STANDARD:
            return StandardScaler()
        else:
            return RobustScaler()

    def get_callbacks(self, model_path: str = None):
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.patience // 2,
                min_lr=1e-7,
                verbose=1
            )
        ]

        if model_path:
            callbacks.append(
                ModelCheckpoint(
                    model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False
                )
            )

        return callbacks


class CNNBiLSTMAttention(BaseModel):

    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        inputs = Input(shape=input_shape)

        # Multi-scale CNN feature extraction
        conv_outputs = []
        for filters, kernel_size in zip(self.config.cnn_filters, self.config.kernel_sizes):
            conv = Conv1D(filters, kernel_size, padding='same', activation='relu')(inputs)
            if self.config.use_batch_norm:
                conv = BatchNormalization()(conv)
            conv = MaxPooling1D(2, padding='same')(conv)
            conv_outputs.append(conv)

        # Concatenate multi-scale features
        if len(conv_outputs) > 1:
            x = Concatenate()(conv_outputs)
        else:
            x = conv_outputs[0]

        # Bidirectional LSTM layers with residual connections
        lstm_outputs = []
        for i, units in enumerate(self.config.lstm_units):
            lstm = Bidirectional(
                LSTM(units, return_sequences=True,
                     kernel_regularizer=l1_l2(self.config.l1_reg, self.config.l2_reg))
            )(x if i == 0 else lstm_outputs[-1])

            if self.config.use_batch_norm:
                lstm = BatchNormalization()(lstm)

            lstm = Dropout(self.config.dropout_rates[min(i, len(self.config.dropout_rates)-1)])(lstm)

            # Residual connection
            if self.config.use_residual and i > 0 and x.shape[-1] == lstm.shape[-1]:
                lstm = Add()([x, lstm])

            lstm_outputs.append(lstm)

        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.config.attention_heads,
            key_dim=lstm_outputs[-1].shape[-1] // self.config.attention_heads
        )(lstm_outputs[-1], lstm_outputs[-1])

        # Combine with LSTM output
        combined = Add()([lstm_outputs[-1], attention_output])
        combined = LayerNormalization()(combined)

        # Flatten and dense layers
        x = Flatten()(combined)

        # Deep dense network
        dense_units = [512, 256, 128, 64]
        for units in dense_units:
            x = Dense(units, activation='relu',
                      kernel_regularizer=l1_l2(self.config.l1_reg, self.config.l2_reg))(x)
            if self.config.use_batch_norm:
                x = BatchNormalization()(x)
            x = Dropout(0.3)(x)

        # Output layer
        outputs = Dense(self.config.future_steps)(x)

        self.model = Model(inputs=inputs, outputs=outputs, name='CNN_BiLSTM_Attention')
        return self.model


class TransformerModel(BaseModel):

    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        inputs = Input(shape=input_shape)

        # Positional encoding
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        position_embedding = Dense(input_shape[1])(tf.expand_dims(positions, 0))
        x = inputs + position_embedding

        # Transformer blocks
        for _ in range(3):
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=self.config.attention_heads,
                key_dim=input_shape[1] // self.config.attention_heads
            )(x, x)

            # Skip connection and normalization
            x = LayerNormalization()(x + attention_output)

            # Feed-forward network
            ffn = Sequential([
                Dense(256, activation='relu'),
                Dropout(0.2),
                Dense(input_shape[1])
            ])
            ffn_output = ffn(x)

            # Skip connection and normalization
            x = LayerNormalization()(x + ffn_output)

        # Global pooling
        x = tf.reduce_mean(x, axis=1)

        # Dense layers
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)

        outputs = Dense(self.config.future_steps)(x)

        self.model = Model(inputs=inputs, outputs=outputs, name='Transformer')
        return self.model


class WaveNetModel(BaseModel):

    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        inputs = Input(shape=input_shape)

        # Initial convolution
        x = Conv1D(64, 1, activation='relu')(inputs)

        skip_connections = []

        # Dilated convolution blocks
        for dilation_rate in [1, 2, 4, 8, 16, 32]:
            # Dilated convolution
            conv = Conv1D(64, 3, dilation_rate=dilation_rate,
                          padding='causal', activation='tanh')(x)

            # Gate
            gate = Conv1D(64, 3, dilation_rate=dilation_rate,
                          padding='causal', activation='sigmoid')(x)

            # Gated activation
            gated = tf.multiply(conv, gate)

            # Skip connection
            skip = Conv1D(64, 1)(gated)
            skip_connections.append(skip)

            # Residual connection
            x = Add()([x, Conv1D(64, 1)(gated)])

        # Combine skip connections
        x = Add()(skip_connections)
        x = tf.nn.relu(x)

        # Output layers
        x = Conv1D(32, 1, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)

        outputs = Dense(self.config.future_steps)(x)

        self.model = Model(inputs=inputs, outputs=outputs, name='WaveNet')
        return self.model


class EnsembleModel:

    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = []
        self.weights = None
        self.scaler = None

    def build_ensemble(self, input_shape: Tuple[int, int]):
        model_classes = [
            CNNBiLSTMAttention,
            TransformerModel,
            WaveNetModel
        ]

        for model_class in model_classes[:self.config.ensemble_models]:
            model_config = ModelConfig(**self.config.to_dict())
            model_instance = model_class(model_config)
            model_instance.build_model(input_shape)
            model_instance.compile_model()
            self.models.append(model_instance)

        # Initialize equal weights
        self.weights = np.ones(len(self.models)) / len(self.models)

        logger.info(f"Built ensemble with {len(self.models)} models")

    def train_ensemble(self, X_train, y_train, X_val, y_val):
        val_scores = []

        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.model.name}")

            history = model.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                callbacks=model.get_callbacks(),
                verbose=1
            )

            # Get validation score
            val_loss = min(history.history['val_loss'])
            val_scores.append(1 / val_loss)  # Inverse for weighting

        # Calculate weights based on validation performance
        val_scores = np.array(val_scores)
        self.weights = val_scores / val_scores.sum()

        logger.info(f"Ensemble weights: {self.weights}")

    def predict_ensemble(self, X):
        predictions = []

        for model, weight in zip(self.models, self.weights):
            pred = model.model.predict(X, verbose=0)
            predictions.append(pred * weight)

        return np.sum(predictions, axis=0)


class ModelFactory:

    @staticmethod
    def create_model(config: ModelConfig) -> Union[BaseModel, EnsembleModel]:
        if config.use_ensemble or config.model_type == ModelType.ENSEMBLE:
            return EnsembleModel(config)

        model_map = {
            ModelType.CNN_BILSTM_ATTENTION: CNNBiLSTMAttention,
            ModelType.TRANSFORMER: TransformerModel,
            ModelType.WAVENET: WaveNetModel,
            ModelType.GRU_ATTENTION: GRUAttentionModel
        }

        model_class = model_map.get(config.model_type, CNNBiLSTMAttention)
        return model_class(config)


class GRUAttentionModel(BaseModel):

    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        inputs = Input(shape=input_shape)

        # GRU layers
        x = inputs
        for i, units in enumerate(self.config.gru_units):
            x = Bidirectional(
                GRU(units, return_sequences=True,
                    kernel_regularizer=l1_l2(self.config.l1_reg, self.config.l2_reg))
            )(x)

            if self.config.use_batch_norm:
                x = BatchNormalization()(x)

            x = Dropout(self.config.dropout_rates[min(i, len(self.config.dropout_rates)-1)])(x)

        # Attention mechanism
        attention = Attention()([x, x])

        # Combine
        x = Concatenate()([x, attention])
        x = Flatten()(x)

        # Dense layers
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)

        outputs = Dense(self.config.future_steps)(x)

        self.model = Model(inputs=inputs, outputs=outputs, name='GRU_Attention')
        return self.model
