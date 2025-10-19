import { NextRequest, NextResponse } from 'next/server';
import connectDB from '@/lib/mongodb';
import Asset from '@/models/Asset';
import { fetchStockPrice } from '@/lib/external-api';

export async function GET(request: NextRequest) {
  try {
    await connectDB();

    const { searchParams } = new URL(request.url);
    const category = searchParams.get('category') || 'stock';
    const limit = parseInt(searchParams.get('limit') || '100');
    const search = searchParams.get('search') || '';

    let query: any = { category };

    if (search) {
      query.$or = [
        { name: { $regex: search, $options: 'i' } },
        { symbol: { $regex: search, $options: 'i' } },
      ];
    }

    const assets = await Asset.find(query).limit(limit).lean();

    const enrichedAssets = await Promise.all(
      assets.map(async (asset) => {
        let currentPrice = asset.currentPrice;

        try {
          currentPrice = await fetchStockPrice(asset.symbol);
        } catch (error) {
          currentPrice = asset.currentPrice;
        }

        return {
          uuid: asset.uuid,
          symbol: asset.symbol,
          name: asset.name,
          category: asset.category,
          currentPrice,
        };
      })
    );

    return NextResponse.json(enrichedAssets, { status: 200 });
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        message: error.message || 'Internal server error',
      },
      { status: 500 }
    );
  }
}
