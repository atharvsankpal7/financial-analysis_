const POLYGON_API_KEY = process.env.POLYGON_API_KEY || '';
const MCX_API_KEY = process.env.MCX_API_KEY || '';

export interface PolygonStockData {
  symbol: string;
  price: number;
  timestamp: number;
}

export interface MCXGoldData {
  state: string;
  price: number;
  date: string;
}

export async function fetchStockPrice(symbol: string): Promise<number> {
  if (!POLYGON_API_KEY) {
    return generateMockStockPrice(symbol);
  }

  try {
    const response = await fetch(
      `https://api.polygon.io/v2/aggs/ticker/${symbol}/prev?adjusted=true&apiKey=${POLYGON_API_KEY}`
    );

    if (!response.ok) {
      return generateMockStockPrice(symbol);
    }

    const data = await response.json();

    if (data.results && data.results.length > 0) {
      return data.results[0].c;
    }

    return generateMockStockPrice(symbol);
  } catch (error) {
    return generateMockStockPrice(symbol);
  }
}

export async function fetchMultipleStockPrices(symbols: string[]): Promise<Record<string, number>> {
  const prices: Record<string, number> = {};

  await Promise.all(
    symbols.map(async (symbol) => {
      prices[symbol] = await fetchStockPrice(symbol);
    })
  );

  return prices;
}

export async function fetchGoldPrice(state: string, date: string): Promise<number> {
  if (!MCX_API_KEY) {
    return generateMockGoldPrice(state);
  }

  try {
    const response = await fetch(
      `https://api.mcxindia.com/gold-price?state=${encodeURIComponent(state)}&date=${date}&apiKey=${MCX_API_KEY}`
    );

    if (!response.ok) {
      return generateMockGoldPrice(state);
    }

    const data = await response.json();

    if (data.price) {
      return data.price;
    }

    return generateMockGoldPrice(state);
  } catch (error) {
    return generateMockGoldPrice(state);
  }
}

function generateMockStockPrice(symbol: string): number {
  const basePrice = 1000;
  const hash = symbol.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const variation = (hash % 5000) + 500;
  return basePrice + variation + Math.random() * 100;
}

function generateMockGoldPrice(state: string): number {
  const basePrice = 6000;
  const hash = state.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const variation = (hash % 500);
  return basePrice + variation + Math.random() * 50;
}

export async function fetchLatestGoldPriceForState(state: string): Promise<{ price: number; date: string }> {
  const today = new Date().toISOString().split('T')[0];
  const price = await fetchGoldPrice(state, today);

  return {
    price,
    date: today,
  };
}

export function getStockSearchResults(query: string, stocks: any[]): any[] {
  if (!query) {
    return stocks;
  }

  const lowerQuery = query.toLowerCase();

  return stocks.filter((stock) => {
    return (
      stock.name.toLowerCase().includes(lowerQuery) ||
      stock.symbol.toLowerCase().includes(lowerQuery)
    );
  });
}
