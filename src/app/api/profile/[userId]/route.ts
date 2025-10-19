import { NextRequest, NextResponse } from "next/server";
import connectDB from "@/lib/mongodb";
import User from "@/models/User";
import UserPortfolio from "@/models/UserPortfolio";
import { initialInfoSchema } from "@/lib/validations";
import { calculateSafeSavings } from "@/lib/utils";
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

    return NextResponse.json(
      {
        success: true,
        data: {
          fullName: user.fullName,
          location: user.location,
          initialInvestmentAmount: user.initialInvestmentAmount,
          savingsThreshold: user.savingsThreshold,
          annualSavingsInterestRate: user.annualSavingsInterestRate,
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

export async function PUT(
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

    const body = await request.json();

    const validationResult = initialInfoSchema.safeParse(body);

    if (!validationResult.success) {
      return NextResponse.json(
        {
          success: false,
          message: "Validation failed",
        },
        { status: 400 },
      );
    }

    const data = validationResult.data;

    if (data.location.country !== "India") {
      return NextResponse.json(
        {
          success: false,
          message: "Service is only available in India",
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

    const oldInvestmentAmount = user.initialInvestmentAmount;
    const oldSafeSavings = calculateSafeSavings(
      oldInvestmentAmount,
      user.savingsThreshold,
    );

    const newInvestmentAmount = data.initialInvestmentAmount;
    const newSafeSavings = calculateSafeSavings(
      newInvestmentAmount,
      data.savingsThreshold,
    );

    user.fullName = data.fullName;
    user.location = data.location;
    user.initialInvestmentAmount = data.initialInvestmentAmount;
    user.savingsThreshold = data.savingsThreshold;
    user.annualSavingsInterestRate = data.annualSavingsInterestRate;

    await user.save();

    if (
      oldInvestmentAmount !== newInvestmentAmount ||
      oldSafeSavings !== newSafeSavings
    ) {
      const portfolio = await UserPortfolio.findOne({ userId });

      if (portfolio) {
        const allocations =
          portfolio.allocations instanceof Map
            ? Object.fromEntries(portfolio.allocations)
            : portfolio.allocations;

        const currentTotal =
          portfolio.savingsAllocation +
          portfolio.goldAllocation +
          Object.values(allocations as Record<string, number>).reduce(
            (sum, val) => sum + val,
            0,
          );

        if (currentTotal !== newInvestmentAmount) {
          const ratio = newInvestmentAmount / currentTotal;

          portfolio.savingsAllocation = Math.max(
            newSafeSavings,
            portfolio.savingsAllocation * ratio,
          );
          portfolio.goldAllocation = portfolio.goldAllocation * ratio;

          const currentAllocations =
            portfolio.allocations instanceof Map
              ? Object.fromEntries(portfolio.allocations)
              : portfolio.allocations;

          const newAllocations: Record<string, number> = {};
          for (const key in currentAllocations as Record<string, number>) {
            newAllocations[key] =
              (currentAllocations as Record<string, number>)[key] * ratio;
          }

          portfolio.allocations = newAllocations as any;

          const newTotal =
            portfolio.savingsAllocation +
            portfolio.goldAllocation +
            Object.values(newAllocations).reduce((sum, val) => sum + val, 0);

          if (Math.abs(newTotal - newInvestmentAmount) > 0.01) {
            const adjustment = newInvestmentAmount - newTotal;
            portfolio.savingsAllocation += adjustment;
          }

          await portfolio.save();
        }
      }
    }

    return NextResponse.json(
      {
        success: true,
        message: "Profile updated successfully",
        data: {
          userId: user._id.toString(),
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
