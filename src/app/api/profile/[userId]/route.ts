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

    const portfolio = await UserPortfolio.findOne({ userId });

    if (portfolio) {
      const allocations =
        portfolio.allocations instanceof Map
          ? Object.fromEntries(portfolio.allocations)
          : portfolio.allocations;

      const currentTotalAllocated =
        portfolio.savingsAllocation +
        portfolio.goldAllocation +
        Object.values(allocations as Record<string, number>).reduce(
          (sum, val) => sum + val,
          0,
        );

      const unallocatedAmount =
        user.initialInvestmentAmount - currentTotalAllocated;
      const investmentReduction =
        user.initialInvestmentAmount - data.initialInvestmentAmount;

      if (investmentReduction > 0 && investmentReduction > unallocatedAmount) {
        return NextResponse.json(
          {
            success: false,
            message: `Cannot reduce investment by ${investmentReduction}. Only ${unallocatedAmount} is unallocated. Please adjust your portfolio allocations first.`,
          },
          { status: 400 },
        );
      }
    }

    const existingUser = await User.findById(userId);

    if (!existingUser) {
      return NextResponse.json(
        {
          success: false,
          message: "User not found",
        },
        { status: 404 },
      );
    }

    const oldInvestmentAmount = existingUser.initialInvestmentAmount;
    const oldSafeSavings = calculateSafeSavings(
      oldInvestmentAmount,
      existingUser.savingsThreshold,
    );

    const newInvestmentAmount = data.initialInvestmentAmount;
    const newSafeSavings = calculateSafeSavings(
      newInvestmentAmount,
      data.savingsThreshold,
    );

    existingUser.fullName = data.fullName;
    existingUser.location = data.location;
    existingUser.initialInvestmentAmount = data.initialInvestmentAmount;
    existingUser.savingsThreshold = data.savingsThreshold;
    existingUser.annualSavingsInterestRate = data.annualSavingsInterestRate;

    await existingUser.save();

    if (
      oldInvestmentAmount !== newInvestmentAmount ||
      oldSafeSavings !== newSafeSavings
    ) {
      const existingPortfolio = await UserPortfolio.findOne({ userId });

      if (existingPortfolio) {
        const allocations =
          existingPortfolio.allocations instanceof Map
            ? Object.fromEntries(existingPortfolio.allocations)
            : existingPortfolio.allocations;

        const currentTotal =
          existingPortfolio.savingsAllocation +
          existingPortfolio.goldAllocation +
          Object.values(allocations as Record<string, number>).reduce(
            (sum, val) => sum + val,
            0,
          );

        if (currentTotal !== newInvestmentAmount) {
          const ratio = newInvestmentAmount / currentTotal;

          existingPortfolio.savingsAllocation = Math.max(
            newSafeSavings,
            existingPortfolio.savingsAllocation * ratio,
          );
          existingPortfolio.goldAllocation =
            existingPortfolio.goldAllocation * ratio;

          const currentAllocations =
            existingPortfolio.allocations instanceof Map
              ? Object.fromEntries(existingPortfolio.allocations)
              : existingPortfolio.allocations;

          const newAllocations: Record<string, number> = {};
          for (const key in currentAllocations as Record<string, number>) {
            newAllocations[key] =
              (currentAllocations as Record<string, number>)[key] * ratio;
          }

          existingPortfolio.allocations = newAllocations as any;

          const newTotal =
            existingPortfolio.savingsAllocation +
            existingPortfolio.goldAllocation +
            Object.values(newAllocations).reduce((sum, val) => sum + val, 0);

          if (Math.abs(newTotal - newInvestmentAmount) > 0.01) {
            const adjustment = newInvestmentAmount - newTotal;
            existingPortfolio.savingsAllocation += adjustment;
          }

          await existingPortfolio.save();
        }
      }
    }

    return NextResponse.json(
      {
        success: true,
        message: "Profile updated successfully",
        data: {
          userId: existingUser._id.toString(),
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
