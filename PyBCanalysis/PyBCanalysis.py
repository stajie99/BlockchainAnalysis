# ---------------------------------------
# print("Hello, this is PyBCanalysis!")
# import sys
# from math import cos, radians

# for i in range(360):
#     print(cos(radians(i)))
# ---------------------------------------

# # install cypo as a py package
# py -m pip install maturin
# py -m pip install cryo
import maturin
import cryo # use as cryo <dataset> [OPTIONS]
# cryo currently obtains all of its data using the JSON-RPC protocol standard.
# 1. set up cryo
# 2. then use the cryo_python library to interact with it. 
# *. cryo is a tool written in Rust that extracts data from blockchains via JSON-RPC requests, 
#     processes it, and saves it in formats like Parquet, CSV, or JSON, or directly to Python 
#     data structures. 

# To use cryo you’ll need a custom RPC (Remote Procedure Call) endpoint.
set ETH_RPC_URL=https://ethereum-mainnet.core.chainstack.com/36d94383e5200effe4a966a88f74e598

cryo blocks --blocks 18734075:18735075 --dry

cryo blocks ^
    --blocks 18734975:18735075 ^
    --csv ^
    --requests-per-second 30 ^
    --columns block_number block_hash timestamp chain_id ^
    --dry


cryo erc20_transfers --blocks latest --json