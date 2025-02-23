from pybit.unified_trading import HTTP
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import ccxt
plt.style.use('fivethirtyeight')
plt.style.use('dark_background')
import asyncio
import nest_asyncio
import pandas as pd
from datetime import datetime, timedelta
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np



nest_asyncio.apply()

async def get_top_volume_symbols(top=10):
    bybit = ccxt.bybit()
    markets = bybit.fetch_markets()
    
    # Lọc các thị trường có loại 'spot', không cần kiểm tra 'USDT'
    spot_markets = [
        market['symbol'] for market in markets 
        if market['type'] == 'spot'
    ]
    
    volumes = []
    for symbol in spot_markets:
        try:
            ticker = bybit.fetch_ticker(symbol)
            volumes.append({
                'symbol': symbol,
                'volume': ticker['quoteVolume'] if ticker['quoteVolume'] else 0
            })
            await asyncio.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"Error fetching volume for {symbol}: {e}")
            continue
    
    # Sắp xếp các cặp giao dịch theo khối lượng giảm dần và lấy top N
    top_symbols = sorted(volumes, key=lambda x: x['volume'], reverse=True)[:top]
    return [v['symbol'] for v in top_symbols]


async def fetch_ohlcv_data():
    bybit = ccxt.bybit()
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=365)
    
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # Lấy top 300 mã có khối lượng giao dịch cao nhất
        all_symbols = await get_top_volume_symbols(400)  # Fetch extra symbols as a buffer
        symbols_to_process = []
        skipped_symbols = []
        
        for symbol in all_symbols:
            if len(symbols_to_process) >= 300:
                break
            
            print(f"\nFetching data for {symbol}")
            since = int(start_time.timestamp() * 1000)
            all_ohlcv = []
            
            try:
                while since < int(end_time.timestamp() * 1000):
                    ohlcv = bybit.fetch_ohlcv(
                        symbol=symbol,
                        timeframe='1d',
                        since=since,
                        limit=1000
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    since = ohlcv[-1][0] + 86400000  # Move to the next day
                    
                    await asyncio.sleep(1)  # Rate limiting
                
                df = pd.DataFrame(
                    all_ohlcv, 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Check if the data covers more than 360 days
                if len(df) >= 360:
                    symbols_to_process.append(symbol)
                    filename = f"data/{symbol.replace('/', '_').replace(':', '_')}_1d_data.csv"
                    df.to_csv(filename, index=False)
                    print(f"Saved data to {filename}")
                else:
                    print(f"Skipping {symbol}, less than 360 days of data available.")
                    skipped_symbols.append(symbol)
            
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                skipped_symbols.append(symbol)
        
        print("\nFinal list of processed symbols:", symbols_to_process)
        print("Skipped symbols:", skipped_symbols)
        
        if len(symbols_to_process) < 300:
            print(f"\nOnly {len(symbols_to_process)} symbols were processed. Consider increasing the buffer size of top volume symbols.")
            
    except Exception as e:
        print(f"Error fetching OHLCV data: {e}")

if __name__ == "__main__":
    asyncio.run(fetch_ohlcv_data())
