import { NextRequest, NextResponse } from "next/server";
import connectDB from "@/lib/mongodb";
import User from "@/models/User";
import { initialInfoSchema } from "@/lib/validations";
import { calculateSafeSavings } from "@/lib/utils";

export async function POST(request: NextRequest) {
  try {
    await connectDB();

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

    const user = await User.create({
      fullName: data.fullName,
      location: data.location,
      initialInvestmentAmount: data.initialInvestmentAmount,
      savingsThreshold: data.savingsThreshold,
      annualSavingsInterestRate: data.annualSavingsInterestRate,
    });

    const safeSavings = calculateSafeSavings(
      data.initialInvestmentAmount,
      data.savingsThreshold,
    );

    return NextResponse.json(
      {
        success: true,
        message: "Profile saved successfully",
        data: {
          userId: user._id.toString(),
          safeSavings,
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
