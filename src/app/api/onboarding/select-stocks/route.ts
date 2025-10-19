import { NextRequest, NextResponse } from "next/server";
import connectDB from "@/lib/mongodb";
import User from "@/models/User";
import UserPortfolio from "@/models/UserPortfolio";
import { selectStocksSchema } from "@/lib/validations";
import mongoose from "mongoose";

export async function POST(request: NextRequest) {
  try {
    await connectDB();

    const body = await request.json();

    const validationResult = selectStocksSchema.safeParse(body);

    if (!validationResult.success) {
      return NextResponse.json(
        {
          success: false,
          message: "Validation failed",
        },
        { status: 400 },
      );
    }

    const { userId, selectedStockIds } = validationResult.data;

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

    const existingPortfolio = await UserPortfolio.findOne({ userId });

    if (existingPortfolio && existingPortfolio.onboardingComplete) {
      return NextResponse.json(
        {
          success: false,
          message: "Onboarding already completed",
        },
        { status: 400 },
      );
    }

    const initialAllocations: Record<string, number> = {};
    selectedStockIds.forEach((uuid) => {
      initialAllocations[uuid] = 0;
    });

    const portfolio = await UserPortfolio.findOneAndUpdate(
      { userId },
      {
        userId,
        selectedStockIds,
        allocations: initialAllocations,
        goldAllocation: 0,
        savingsAllocation: 0,
        onboardingComplete: true,
      },
      { upsert: true, new: true },
    );

    return NextResponse.json(
      {
        success: true,
        message: "Portfolio initialized",
        data: {
          portfolioId: portfolio._id.toString(),
        },
      },
      { status: 201 },
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
