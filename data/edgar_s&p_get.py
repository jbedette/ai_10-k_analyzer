from sec_edgar_downloader import Downloader

ticker_symbols = [
  "MMM", "ABT", "ABBV", "ACN", "ATVI", "ADM", "ADBE", "ADP", "AAP", "AES",
  "AFL", "A", "APD", "AKAM", "ALK", "ALB", "ARE", "ALGN", "ALLE", "LNT",
  "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP", "AXP",
  "AIG", "AMT", "AWK", "AMP", "ABC", "AME", "AMGN", "APH", "ADI", "ANSS",
  "AON", "APA", "AAPL", "AMAT", "APTV", "ANET", "AJG", "AIZ", "T", "ATO",
  "ADSK", "AZO", "AVB", "AVY", "BKR", "BALL", "BAC", "BBWI", "BAX", "BDX",
  "WRB", "BRK.B", "BBY", "BIO", "TECH", "BIIB", "BLK", "BK", "BA", "BKNG",
  "BWA", "BXP", "BSX", "BMY", "AVGO", "BR", "BRO", "BF.B", "CHRW", "CDNS",
  "CZR", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CTLT", "CAT", "CBOE",
  "CBRE", "CDW", "CE", "CNC", "CNP", "CDAY", "CERN", "CF", "CRL", "SCHW",
  "CHTR", "CVX", "CMG", "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C",
  "CFG", "CTXS", "CLX", "CME", "CMS", "KO", "CL", "CMCSA", "CMA", "CAG",
  "COP", "ED", "STZ", "CEG", "COO", "CPRT", "GLW", "CTVA", "COST", "CTRA",
  "CCI", "CSX", "CMI", "CVS", "DHI", "DHR", "DRI", "DVA", "DE", "DAL",
  "XRAY", "DVN", "DXCM", "FANG", "DLR", "DFS", "DISCA", "DISCK", "DISH",
  "DIS", "DG", "DLTR", "D", "DPZ", "DOV", "DOW", "DTE", "DUK", "DRE",
  "DD", "DXC", "EMN", "ETN", "EBAY", "ECL", "EIX", "EW", "EA", "EMR",
  "ENPH", "ETR", "EOG", "EPAM", "EFX", "EQIX", "EQR", "ESS", "EL", "ETSY",
  "RE", "EVRG", "ES", "EXC", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FDS",
  "FAST", "FRT", "FDX", "FITB", "FRC", "FE", "FIS", "FISV", "FLT", "FMC",
  "F", "FTNT", "FTV", "FBHS", "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT",
  "GE", "GNRC", "GD", "GIS", "GM", "GPC", "GILD", "GL", "GPN", "GS",
  "HAL", "HBI", "HIG", "HAS", "HCA", "PEAK", "HSIC", "HSY", "HES", "HPE",
  "HLT", "HOLX", "HD", "HON", "HRL", "HST", "HWM", "HPQ", "HUM", "HBAN",
  "HII", "IBM", "IEX", "IDXX", "INFO", "ITW", "ILMN", "INCY", "IR", "INTC",
  "ICE", "IP", "IPG", "IFF", "INTU", "ISRG", "IVZ", "IPGP", "IQV", "IRM",
  "JBHT", "JKHY", "J", "SJM", "JNJ", "JCI", "JPM", "JNPR", "K", "KEY",
  "KEYS", "KMB", "KIM", "KMI", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX",
  "LW", "LVS", "LDOS", "LEN", "LLY", "LNC", "LIN", "LYV", "LKQ", "LMT",
  "L", "LOW", "LUMN", "LYB", "MTB", "MRO", "MPC", "MKTX", "MAR", "MMC",
  "MLM", "MAS", "MA", "MTCH", "MKC", "MCD", "MCK", "MDT", "MRK", "META",
  "MET", "MTD", "MGM", "MCHP", "MU", "MSFT", "MAA", "MRNA", "MHK", "TAP",
  "MDLZ", "MPWR", "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ",
  "NTAP", "NFLX", "NWL", "NEM", "NWSA", "NWS", "NEE", "NLSN", "NKE",
  "NI", "NDSN", "NSC", "NTRS", "NOC", "NLOK", "NCLH", "NOV", "NRG",
  "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "OKE",
  "ORCL", "OGN", "OTIS", "PCAR", "PKG", "PARA", "PH", "PAYX", "PAYC",
  "PYPL", "PENN", "PNR", "PBCT", "PEP", "PKI", "PRGO", "PFE", "PM",
  "PSX", "PNW", "PXD", "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR",
  "PLD", "PRU", "PTC", "PEG", "PSA", "PHM", "PVH"]
 

invalid = []

dl = Downloader("./10-k_reports", "bedette.john@gmail.com")
for symbol in ticker_symbols:
    print(f"{symbol}")
    try:
        dl.get("10-K", symbol)  # Replace 'AAPL' with the ticker symbol
    except ValueError as e:
        invalid.append(symbol)
        print(f"skipping {symbol} due to err: {e}")
    print(invalid)
