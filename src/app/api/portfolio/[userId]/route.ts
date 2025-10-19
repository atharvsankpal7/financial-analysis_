import { NextRequest, NextResponse } from "next/server";
import connectDB from "@/lib/mongodb";
import User from "@/models/User";
import UserPortfolio from "@/models/UserPortfolio";
import Asset from "@/models/Asset";
import GoldPrice from "@/models/GoldPrice";
import { calculateTotalValue, calculateDistribution } from "@/lib/utils";
import { calculatePredictedReturns } from "@/lib/predictions";
import { fetchStockPrice } from "@/lib/external-api";
import mongoose from "mongoose";

export async function GET(
  request: NextRequest,
  { params }: { params: { userId: string } },
) {
  try {
    await connectDB();

    const { userId } = params;

    if (!mongoose.Types.ObjectId.isValid(userId)) {
      return NextResponse.json(
        {
          success: false,
          message: "Invalid user ID",
        },
        { status: 400 },
      );
    }

    const user = await User.findById(userId);

    if (!user) {
      return NextResponse.json(
        {
          success: false,
          message: "User not found",
        },
        { status: 404 },
      );
    }

    const portfolio = await UserPortfolio.findOne({ userId });

    if (!portfolio) {
      return NextResponse.json(
        {
          success: false,
          message: "Portfolio not found",
        },
        { status: 404 },
      );
    }

    const stockAssets = await Asset.find({
      uuid: { $in: portfolio.selectedStockIds },
    }).lean();

    const enrichedStocks = await Promise.all(
      stockAssets.map(async (asset) => {
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
      }),
    );

    const latestGoldPrice = await GoldPrice.findOne({
      state: user.location.state,
    })
      .sort({ date: -1 })
      .lean();

    const goldPrice = latestGoldPrice?.price || 6000;

    const allocations =
      portfolio.allocations instanceof Map
        ? Object.fromEntries(portfolio.allocations)
        : portfolio.allocations;

    const totalValue = calculateTotalValue(
      allocations,
      portfolio.goldAllocation,
      portfolio.savingsAllocation,
    );

    const distribution = calculateDistribution(
      allocations,
      portfolio.goldAllocation,
      portfolio.savingsAllocation,
    );

    const predictedReturns = calculatePredictedReturns(
      allocations,
      portfolio.goldAllocation,
      portfolio.savingsAllocation,
      user.annualSavingsInterestRate,
    );

    const unallocatedAmount = user.initialInvestmentAmount - totalValue;

    return NextResponse.json(
      {
        success: true,
        data: {
          portfolio: {
            userId: portfolio.userId.toString(),
            allocations,
            goldAllocation: portfolio.goldAllocation,
            savingsAllocation: portfolio.savingsAllocation,
            totalValue,
            distribution,
            unallocatedAmount,
          },
          marketData: {
            stocks: enrichedStocks,
            goldPrice,
          },
          predictedReturns,
        },
      },
      { status: 200 },
    );
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        message: error.message || "Internal server error",
      },
      { status: 500 },
    );
  }
}
