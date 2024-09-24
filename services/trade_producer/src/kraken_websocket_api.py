from typing import List
from websocket import create_connection
from loguru import logger
import json

from pydantic import BaseModel

class Trade(BaseModel):
    product_id: str
    quantity: float
    price: float
    timestamp_ms: int

class KrakenWebSocketAPI:
    """
    Class for reading real_time trades from the Kraken Websocket API
    """
    URL = 'wss://ws.kraken.com/v2'
    
    
    
    def __init__(self, product_id: str):
        """
        Initializes the KrakenWebsocketAPI instance

        Args:
            product_id (str): The ptoduct id to get the trades from Kraken WebsocketAPI
        """

        self.product_id = product_id
        
        # establish connection to the Kraken websocket API
        self._ws = create_connection(self.URL)
        logger.debug("Connection Esablished")
        
        # Subscribe to the trades for the given "product_id"
       
        self._subscribe(product_id)
        
    def get_trades(self) -> List[Trade]:
        """
        Returns:
            List[dict]: Returns the latest batch of trades from the Kraken WebSocket API
        """
        
        message = self._ws.recv()
        
        if 'heartbeat' in message:
            # when I get a heartbeat, I return an empty list
            logger.debug('Heartbeat Received')
            return []
        
        # parse the message string as a dictionary
        message = json.loads(message)
        
        # extract trades from the message['data'] field
        trades = []
        for trade in message['data']:
            # extract the following fields:
            # -product id
            # -price
            # -qty
            # -timestamp
            # breakpoint()
            trades.append(
                Trade(
                    product_id=trade['symbol'],
                    quantity=trade['qty'],
                    price=trade['price'],
                    timestamp_ms=self.to_ms(trade['timestamp']),
                )
            )
        return trades
        
    
    def is_done(self) -> bool:
        """
        Returns:
            bool: Returns True if Kraken API connection is closed
        """
        False
    
    

    def _subscribe(self, product_id: str):
        """
        Establish connection to the Kraken websocket API and subscribe to the trades for the given 'product_id

        Args:
            product_ids (List[str]): symbol of the product that we want to subscribe
            
        """
        logger.info(f"Subscribing to trades for {product_id}")
        # logger.info(f"Subscribing to trades for {product_ids}")
        
        # Let's subscribe to the trades for the given 'product_id'
        
        # for product_id in product_id:
        msg = {
            "method": "subscribe",
            "params": {
                "channel": "trade",
                "symbol": [product_id],
                "snapshot": False,
                },
            }
        self._ws.send(json.dumps(msg))
        logger.info(f'Subscription for {product_id} worked!')
    
        # For each product_id we dump
        # The first 2 messages we get from the websocket, because they contain
        # no trade data, just confirmation on their end that the subscription was successful
    
        for product_id in [product_id]:
            _ = self._ws.recv()
            _ = self._ws.recv()
            
    @staticmethod
    def to_ms(timestamp: str) -> int:
        """
        A function that transforms a timestamp expressed
        as a string into a timestamp expressed in milliseconds.

        Args:
            timestamp (str): A timestamp expressed as a string.

        Returns:
            int: A timestamp expressed in milliseconds.
            
        """
        from datetime import datetime, timezone

        timestamp = datetime.fromisoformat(timestamp[:-1]).replace(tzinfo=timezone.utc)
        return int(timestamp.timestamp()*1000)