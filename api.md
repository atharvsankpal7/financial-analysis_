# Detailed Data Flow and API Specifications

This document provides a comprehensive overview of the data flow across the financial investment application's onboarding flow and investment adjustment interface. It builds on the finalized SRS by detailing the end-to-end data movement, including client-server interactions, database operations, and external integrations. The focus is on APIs (using Next.js Route Handlers or Server Actions), their inputs (request body/query params), actions (validation, processing, persistence), and outputs (response body/status).

The architecture ensures:
- **Security**: Authentication via JWT/session (assumed middleware on all APIs).
- **Validation**: Client-side (real-time) + server-side (e.g., Zod schemas).
- **Error Handling**: Standardized responses (e.g., { success: boolean, message: string, data?: any }).
- **Real-Time Aspects**: WebSockets or polling for market data (not detailed here; use libraries like SWR/React Query).
- **Database**: MongoDB with Mongoose schemas for `users`, `assets`, `userPortfolios`, `goldPrices`.

Data flows sequentially through phases, with asynchronous elements (e.g., cron jobs) for maintenance.

## Overall Data Flow Diagram (Conceptual)
```
Client (React/Next.js) → Onboarding Context (Zustand/Context) → API Calls → Server (Next.js) → MongoDB
↑ Real-time Updates (SWR/React Query)                  ↓ External APIs (Stocks/Gold)
↓ Dashboard/Adjustment UI ← Predictions (Server Calc) ← Cron Jobs (Daily Updates)
```

## 1. Onboarding – Initial Information

### Data Flow
1. **Client Load**: Page mounts → Trigger browser Geolocation API (client-side) → Autofill location in form state.
2. **User Input**: Form fields update local state (e.g., React Hook Form) with real-time validation.
3. **Submission**: On "Continue" → Validate form → Send to API → Server persists to `users` → Redirect to Step 2.
4. **Temporary Storage**: Use Zustand/Context for multi-step continuity (e.g., pass userId to next step).

### APIs Involved
#### POST /api/onboarding/initial-info
- **Purpose**: Create or update initial user profile in `users` collection.
- **Input (Request Body)**:
  ```json
  {
    "fullName": "string",  // e.g., "John Doe"
    "location": {
      "state": "string",   // e.g., "Maharashtra"
      "city": "string",    // e.g., "Mumbai"
      "coordinates": { "lat": number, "lng": number }  // Optional, from Geolocation
    },
    "initialInvestmentAmount": number,  // e.g., 100000 (INR)
    "savingsThreshold": {
      "type": "percentage" | "fixed",  // e.g., "percentage"
      "value": number  // e.g., 20 (for 20%)
    },
    "annualSavingsInterestRate": number  // e.g., 6.5 (%)
  }
  ```
- **Actions**:
  1. Authenticate user (from session).
  2. Validate input (e.g., amount > 0, rate 0-10%, location India-only).
  3. Calculate safe savings if needed (e.g., threshold * initialInvestment).
  4. Create/update `users` document (upsert by userId).
  5. Log analytics (optional).
- **Output (Response)**:
  - **Success (200)**:
    ```json
    {
      "success": true,
      "message": "Profile saved successfully",
      "data": { "userId": "ObjectId", "safeSavings": number }  // Derived safe savings
    }
    ```
  - **Error (400/401)**: `{ "success": false, "message": "Invalid investment amount" }`

## 2. Onboarding – Investment Selection

### Data Flow
1. **Client Load**: Fetch available stocks from backend → Render searchable list in local state.
2. **User Interaction**: Search/select stocks → Update local array of UUIDs.
3. **Submission**: On "Finish Setup" → Validate (at least 1 stock) → Send UUIDs to API → Server creates `userPortfolios` → Set onboarding complete → Redirect to dashboard.
4. **Context Carryover**: Use userId from previous step.

### APIs Involved
#### GET /api/assets/stocks (Pre-load for UI)
- **Purpose**: Fetch list of investable stocks for selection.
- **Input (Query Params)**: None (or optional `?category=stock&limit=100`).
- **Actions**:
  1. Query `assets` collection (filter by category="stock").
  2. Enrich with currentPrice from external API (e.g., Polygon.io stocks endpoint).
  3. Cache in Redis (optional for performance).
- **Output (Response)**:
  - **Success (200)**:
    ```json
    [
      {
        "uuid": "string",
        "symbol": "string",  // e.g., "TCS.NS"
        "name": "string",    // e.g., "Tata Consultancy Services"
        "category": "stock",
        "currentPrice": number  // e.g., 3500.50
      }
    ]
    ```

#### POST /api/onboarding/select-stocks
- **Purpose**: Associate selected stocks with user portfolio.
- **Input (Request Body)**:
  ```json
  {
    "userId": "ObjectId",
    "selectedStockIds": ["string", ...]  // Array of UUIDs, e.g., ["uuid1", "uuid2"]
  }
  ```
- **Actions**:
  1. Validate userId exists and onboarding incomplete.
  2. Create new `userPortfolios` document with initial allocations (all 0).
  3. Set `selectedStockIds`, `onboardingComplete: true`.
  4. Trigger initial prediction calc (optional, via queue).
- **Output (Response)**:
  - **Success (201)**:
    ```json
    {
      "success": true,
      "message": "Portfolio initialized",
      "data": { "portfolioId": "ObjectId" }
    }
    ```
  - **Error (400)**: `{ "success": false, "message": "No stocks selected" }`

## 3. Running Phase – Data Display (Portfolio Visualization)

### Data Flow
1. **Client Load**: Server Components fetch initial portfolio/market data → Render dashboard.
2. **Real-Time Updates**: Client polls external APIs (stocks) and queries DB (gold) every 30s-1min.
3. **Derived Calcs**: Client-side: Total value, % allocation, daily change (using fetched prices).
4. **Cron Integration**: Daily server cron fetches/updates gold prices → No direct client impact (query latest).

### APIs Involved
#### GET /api/portfolio/[userId]
- **Purpose**: Fetch user's full portfolio for dashboard display.
- **Input (Path Params)**: `userId` (ObjectId).
- **Actions**:
  1. Query `userPortfolios` by userId.
  2. Join/enrich `assets` for stock details.
  3. Fetch real-time prices (external: Polygon.io for stocks).
  4. Calculate derived: totalValue, allocations %.
  5. Include pre-calculated predicted returns (from snapshot).
- **Output (Response)**:
  - **Success (200)**:
    ```json
    {
      "success": true,
      "data": {
        "portfolio": {
          "userId": "ObjectId",
          "allocations": { "uuid1": number, ... },  // e.g., { "uuid1": 50000 }
          "goldAllocation": number,
          "savingsAllocation": number,
          "totalValue": number,  // Live calc
          "distribution": { "savings": 30, "gold": 20, "stocks": 50 }  // %
        },
        "marketData": { "stocks": [{ "uuid": "str", "currentPrice": num, ... }], "goldPrice": num },
        "predictedReturns": { "stocks": { "uuid1": 12.5 }, "gold": 8.0, "savings": 6.5 }  // %
      }
    }
    ```

#### GET /api/gold-prices/latest?state=string
- **Purpose**: Fetch latest state-specific gold price (supports cron-updated data).
- **Input (Query Params)**: `state` (e.g., "Maharashtra").
- **Actions**: Query `goldPrices` collection (latest by date/state).
- **Output (Response)**: `{ "success": true, "data": { "price": number, "date": "YYYY-MM-DD" } }` (200).

**Cron Job (Server-Side, Non-API)**: Daily at 6 AM IST → Fetch from external (e.g., MCX API) → Upsert `goldPrices` by state/date.

## 4. Running Phase – Prediction & Adjustment

### Data Flow
1. **Trigger**: "Adjust Investments" click → Open modal → Fetch current portfolio/predictions.
2. **Client-Side**: Local state for sliders/proposals → Real-time calc remainingDisposable.
3. **Submission**: Validate (sum == disposable) → Send proposals → Server updates + recalcs predictions.
4. **Post-Update**: Refresh dashboard data.

### APIs Involved
#### GET /api/portfolio/[userId]/predictions (Pre-load for modal)
- **Purpose**: Fetch current allocations and server-calculated predictions.
- **Input (Path Params)**: `userId`.
- **Actions**:
  1. Fetch from `userPortfolios`.
  2. Run prediction engine (server-side: historical avg/custom model via e.g., simple regression or ML lib).
  3. Calc disposableAmount = totalInvestment - safeSavings.
- **Output (Response)**:
  - **Success (200)**:
    ```json
    {
      "success": true,
      "data": {
        "disposableAmount": number,
        "currentAllocations": { "uuid1": number, "gold": number },
        "predictedReturns": { "uuid1": 12.5, "gold": 8.0, "savings": 6.5 }
      }
    }
    ```

#### PUT /api/portfolio/[userId]/adjust
- **Purpose**: Apply proposed rebalancing.
- **Input (Request Body)**:
  ```json
  {
    "proposedAllocations": { "uuid1": number, "uuid2": number, "gold": number }  // New amounts
  }
  ```
- **Actions**:
  1. Validate: Sum(proposed) == disposableAmount, positive values.
  2. Simulate transaction (e.g., check liquidity, but simplified here).
  3. Update `userPortfolios.allocations`.
  4. Recalc predictions (store snapshot in `portfolioSnapshots` collection).
  5. Trigger email/notification (optional).
- **Output (Response)**:
  - **Success (200)**:
    ```json
    {
      "success": true,
      "message": "Portfolio adjusted",
      "data": { "newTotalValue": number, "updatedPredictions": { ... } }
    }
    ```
  - **Error (400)**: `{ "success": false, "message": "Allocations exceed disposable amount" }`

## Additional System Flows
- **External Integrations**:
  - Stocks: Polygon.io API (key in env) → Called in GET /api/assets/stocks and /api/portfolio/[userId].
  - Gold: MCX or similar → Cron-only.
- **Error/Edge Cases**: All APIs return 500 on DB failure; client shows toasts.
- **Scalability**: Use queues (e.g., BullMQ) for predictions; index MongoDB on userId/uuid.
- **Analytics**: Log all API calls to `auditLogs` collection for compliance.

This ensures a seamless, secure flow from onboarding to ongoing management. For implementation, refer to Next.js docs for Route Handlers.
