# Authentication & Onboarding API Documentation

This document describes the separation of login data and onboarding data in the backend API.

## Overview

The authentication and onboarding flow has been refactored into two distinct phases:

1. **Authentication Phase**: User signs up with login credentials only (email/password)
2. **Onboarding Phase**: User completes profile information and stock selection after login

## Flow Diagram

```
┌─────────────┐
│   Sign Up   │ → Saves: email, password only
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Sign In   │ → Authenticates user, creates session
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Check Status │ → Determines if onboarding is complete
└──────┬──────┘
       │
       ├─── Onboarding Complete? ──→ Redirect to Dashboard
       │
       └─── Onboarding Incomplete ──┐
                                     │
                                     ▼
                           ┌──────────────────┐
                           │ Profile Info     │ → Saves: name, location, investment details
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │ Stock Selection  │ → Saves: selected stocks, marks onboarding complete
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │    Dashboard     │
                           └──────────────────┘
```

## API Endpoints

### 1. Sign Up (Registration)

**Endpoint**: `POST /api/auth/signup`

**Purpose**: Create a new user account with login credentials only.

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

**Success Response** (201):
```json
{
  "success": true,
  "message": "Account created successfully",
  "data": {
    "userId": "507f1f77bcf86cd799439011",
    "email": "user@example.com"
  }
}
```

**Error Responses**:
- 400: Validation failed or email already exists
- 500: Internal server error

**What it does**:
- Validates email and password
- Checks if email already exists
- Hashes password with bcrypt
- Creates user document with only email and password
- Creates empty portfolio document for the user
- Does NOT save any profile/onboarding data

---

### 2. Sign In (Login)

**Endpoint**: `POST /api/auth/signin` (via NextAuth)

**Purpose**: Authenticate user and create session.

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

**Success Response**:
- Creates NextAuth session
- Returns session token
- Redirects to appropriate page based on onboarding status

**What it does**:
- Verifies email and password
- Creates JWT session token
- Sets secure HTTP-only cookies

---

### 3. Check Auth Status

**Endpoint**: `GET /api/auth/status`

**Purpose**: Check if user is authenticated and if onboarding is complete.

**Authentication**: Requires valid session (JWT token)

**Success Response** (200):
```json
{
  "success": true,
  "data": {
    "isAuthenticated": true,
    "hasCompletedOnboarding": true,
    "hasProfileInfo": true,
    "hasStockSelection": true,
    "userId": "507f1f77bcf86cd799439011",
    "email": "user@example.com",
    "fullName": "John Doe"
  }
}
```

**Unauthenticated Response** (401):
```json
{
  "success": false,
  "message": "Unauthorized",
  "data": {
    "isAuthenticated": false,
    "hasCompletedOnboarding": false
  }
}
```

**What it checks**:
- `hasProfileInfo`: fullName, location, initialInvestmentAmount, savingsThreshold, annualSavingsInterestRate are all set
- `hasStockSelection`: portfolio exists with at least one selected stock
- `hasCompletedOnboarding`: both profile info and stock selection are complete AND portfolio.onboardingComplete is true

---

### 4. Onboarding - Initial Profile Info

**Endpoint**: `POST /api/onboarding/initial-info`

**Purpose**: Save user's profile and investment information.

**Authentication**: Requires valid session (JWT token)

**Request Body**:
```json
{
  "fullName": "John Doe",
  "location": {
    "state": "Maharashtra",
    "city": "Mumbai",
    "country": "India",
    "coordinates": {
      "lat": 19.076,
      "lng": 72.8777
    }
  },
  "initialInvestmentAmount": 100000,
  "savingsThreshold": {
    "type": "percentage",
    "value": 20
  },
  "annualSavingsInterestRate": 6.5
}
```

**Success Response** (200):
```json
{
  "success": true,
  "message": "Profile saved successfully",
  "data": {
    "userId": "507f1f77bcf86cd799439011",
    "safeSavings": 20000
  }
}
```

**Error Responses**:
- 401: Unauthorized (not logged in)
- 400: Validation failed or service not available in country
- 404: User not found
- 500: Internal server error

**What it does**:
- Validates all profile fields
- Checks that country is India
- Updates user document with profile information
- Ensures portfolio document exists
- Calculates safe savings amount based on threshold
- Does NOT mark onboarding as complete (user must still select stocks)

---

### 5. Onboarding - Stock Selection

**Endpoint**: `POST /api/onboarding/select-stocks`

**Purpose**: Save user's initial stock selections and complete onboarding.

**Authentication**: Requires valid session (JWT token)

**Request Body**:
```json
{
  "selectedStockIds": [
    "550e8400-e29b-41d4-a716-446655440001",
    "550e8400-e29b-41d4-a716-446655440002",
    "550e8400-e29b-41d4-a716-446655440003"
  ]
}
```

**Success Response** (200):
```json
{
  "success": true,
  "message": "Stock selection saved successfully. Onboarding complete!",
  "data": {
    "portfolioId": "507f1f77bcf86cd799439012",
    "userId": "507f1f77bcf86cd799439011",
    "selectedStockIds": [
      "550e8400-e29b-41d4-a716-446655440001",
      "550e8400-e29b-41d4-a716-446655440002",
      "550e8400-e29b-41d4-a716-446655440003"
    ],
    "onboardingComplete": true
  }
}
```

**Error Responses**:
- 401: Unauthorized (not logged in)
- 400: Validation failed, profile info not complete, or onboarding already completed
- 404: User not found
- 500: Internal server error

**What it does**:
- Validates that user has completed profile information first
- Checks that onboarding is not already complete
- Initializes allocations for selected stocks (all set to 0 initially)
- Updates portfolio with selected stock IDs
- Sets `onboardingComplete: true` in portfolio
- User can now access the dashboard

---

## Data Models

### User Model (Updated)

```typescript
interface IUser {
  email: string;              // Required (login data)
  password?: string;          // Required (login data, select: false)
  fullName?: string;          // Optional (onboarding data)
  location?: {                // Optional (onboarding data)
    state: string;
    city: string;
    coordinates?: {
      lat: number;
      lng: number;
    };
    country: string;
  };
  initialInvestmentAmount?: number;           // Optional (onboarding data)
  savingsThreshold?: {                        // Optional (onboarding data)
    type: "percentage" | "fixed";
    value: number;
  };
  annualSavingsInterestRate?: number;         // Optional (onboarding data)
  createdAt: Date;
  updatedAt: Date;
}
```

### UserPortfolio Model (Unchanged)

```typescript
interface IUserPortfolio {
  userId: ObjectId;                    // Reference to User
  selectedStockIds: string[];          // Array of stock UUIDs
  allocations: Record<string, number>; // Map of stock UUID to allocation amount
  goldAllocation: number;              // Amount allocated to gold
  savingsAllocation: number;           // Amount allocated to savings
  onboardingComplete: boolean;         // True when stock selection is done
  createdAt: Date;
  updatedAt: Date;
}
```

---

## Authentication Flow

### Session Management

- Uses NextAuth with JWT strategy
- Session token stored in HTTP-only cookie
- Token includes user ID for server-side lookups
- Session validated on every protected API call

### Protected Routes

All onboarding and dashboard routes require authentication:
- Check session using `getServerSession(authOptions)`
- Return 401 if no valid session
- Extract `session.user.id` for database queries

---

## Validation Rules

### Sign Up
- Email: Must be valid email format
- Password: Minimum 6 characters

### Profile Info (Onboarding Step 1)
- Full Name: 2-100 characters
- State: Required, non-empty string
- City: Required, non-empty string
- Country: Must be "India"
- Initial Investment: Minimum ₹1,000
- Savings Threshold Type: "percentage" or "fixed"
- Savings Threshold Value: >= 0
- Annual Savings Interest Rate: 0-100

### Stock Selection (Onboarding Step 2)
- Selected Stock IDs: Array of strings, minimum 1 stock
- Must have completed profile info first
- Cannot select stocks if onboarding already complete

---

## Error Handling

All endpoints return consistent error responses:

```json
{
  "success": false,
  "message": "Error description",
  "errors": [/* Validation errors if applicable */]
}
```

Common error codes:
- 400: Bad request (validation failed)
- 401: Unauthorized (not logged in)
- 404: Resource not found
- 500: Internal server error

---

## Frontend Integration Notes

### Recommended Flow

1. **Sign Up Page**:
   - POST to `/api/auth/signup` with email/password
   - On success, automatically sign in the user

2. **Sign In Page**:
   - POST to `/api/auth/signin` (NextAuth)
   - On success, check auth status

3. **Auth Status Check**:
   - GET `/api/auth/status` on app load or after login
   - If `hasCompletedOnboarding === false`, redirect to onboarding
   - If `hasProfileInfo === false`, start at step 1 (profile info)
   - If `hasStockSelection === false`, start at step 2 (stock selection)
   - If both complete, redirect to dashboard

4. **Onboarding Step 1**:
   - POST to `/api/onboarding/initial-info`
   - On success, move to step 2 (stock selection)

5. **Onboarding Step 2**:
   - POST to `/api/onboarding/select-stocks`
   - On success, redirect to dashboard

### Session Handling

```typescript
// Example: Check if user needs onboarding
const response = await fetch('/api/auth/status');
const { data } = await response.json();

if (!data.isAuthenticated) {
  // Redirect to sign in
  router.push('/auth/signin');
} else if (!data.hasCompletedOnboarding) {
  if (!data.hasProfileInfo) {
    // Redirect to profile info
    router.push('/onboarding/initial-info');
  } else if (!data.hasStockSelection) {
    // Redirect to stock selection
    router.push('/onboarding/select-stocks');
  }
} else {
  // User has completed onboarding
  router.push(`/dashboard/${data.userId}`);
}
```

---

## Security Considerations

1. **Password Security**:
   - Passwords hashed with bcrypt (10 rounds)
   - Never returned in API responses
   - Stored with `select: false` in schema

2. **Session Security**:
   - HTTP-only cookies prevent XSS attacks
   - JWT tokens signed with secret
   - Session expires after inactivity

3. **Authorization**:
   - All onboarding endpoints check session
   - User can only update their own data
   - User ID from session, not from request body

4. **Validation**:
   - All inputs validated with Zod schemas
   - Sanitization of user inputs
   - Type safety with TypeScript

---

## Testing

### Manual Testing Flow

1. Create new account: `POST /api/auth/signup`
2. Sign in: `POST /api/auth/signin`
3. Check status: `GET /api/auth/status` (should show onboarding incomplete)
4. Add profile info: `POST /api/onboarding/initial-info`
5. Check status: `GET /api/auth/status` (should show stock selection needed)
6. Select stocks: `POST /api/onboarding/select-stocks`
7. Check status: `GET /api/auth/status` (should show onboarding complete)
8. Access dashboard with user ID from status response

### Test Data

```json
// Sign up
{
  "email": "test@example.com",
  "password": "test123456"
}

// Profile info
{
  "fullName": "Test User",
  "location": {
    "state": "Maharashtra",
    "city": "Mumbai",
    "country": "India"
  },
  "initialInvestmentAmount": 50000,
  "savingsThreshold": {
    "type": "percentage",
    "value": 20
  },
  "annualSavingsInterestRate": 6.5
}

// Stock selection (use actual stock UUIDs from your database)
{
  "selectedStockIds": [
    "actual-uuid-1",
    "actual-uuid-2",
    "actual-uuid-3"
  ]
}
```

---

## Migration Notes

### Changes from Previous Version

1. **User Model**: 
   - Made onboarding fields optional (fullName, location, etc.)
   - Only email and password required at signup

2. **Signup Route**:
   - Removed default values for onboarding fields
   - Creates empty portfolio automatically
   - Only saves login credentials

3. **Onboarding Routes**:
   - Added session authentication
   - Removed userId from request body (uses session)
   - Better validation of onboarding completion state

4. **New Endpoint**:
   - `/api/auth/status` to check onboarding progress

### Database Migration

If you have existing users in the database, they should continue to work as all onboarding fields are now optional. New users will go through the two-phase flow.

---

## Summary

The refactored authentication and onboarding system provides:

✅ **Clear separation** between login data and onboarding data  
✅ **Session-based authentication** with NextAuth and JWT  
✅ **Progressive onboarding** with status tracking  
✅ **Security best practices** with bcrypt and HTTP-only cookies  
✅ **Flexible data model** that supports both new and existing users  
✅ **Robust validation** at every step  
✅ **Better user experience** with step-by-step guidance  

The backend is now properly structured with minimal changes needed on the frontend.