import { NextRequest, NextResponse } from "next/server";
import connectDB from "@/lib/mongodb";
import GoldPrice from "@/models/GoldPrice";
import { goldPriceQuerySchema } from "@/lib/validations";

export async function GET(request: NextRequest) {
  try {
    await connectDB();

    const { searchParams } = new URL(request.url);
    const state = searchParams.get("state");

    const validationResult = goldPriceQuerySchema.safeParse({ state });

    if (!validationResult.success) {
      return NextResponse.json(
        {
          success: false,
          message: "State parameter is required",
        },
        { status: 400 },
      );
    }

    const latestGoldPrice = await GoldPrice.findOne({
      state: validationResult.data.state,
    })
      .sort({ date: -1 })
      .lean();

    if (!latestGoldPrice) {
      return NextResponse.json(
        {
          success: false,
          message: "No gold price data found for this state",
        },
        { status: 404 },
      );
    }

    return NextResponse.json(
      {
        success: true,
        data: {
          price: latestGoldPrice.price,
          date: latestGoldPrice.date,
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
