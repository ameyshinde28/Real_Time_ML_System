from abc import ABC, abstractmethod
from typing import List

from trade_data_source.trade import Trade

class TradeSource(ABC):
    @abstractmethod
    def get_trades(self) -> List[Trade]:
        """
        Retrive  the trades from the data source
        """
        pass
    
    @abstractmethod
    def is_done(self) -> bool:
        """
        Return True if there are no more trades to retrieve from the data source, False otherwise
        """
        pass
