# Frontend Migration Guide - Auth & Onboarding Flow

This guide explains the minimal frontend changes needed to work with the new separated authentication and onboarding backend.

## Overview of Changes

The backend now separates:
1. **Login data** (email/password) - saved at signup
2. **Onboarding data** (profile info, stock selection) - saved after login

## Key Backend Changes

### What Changed
- ✅ Signup now only saves email and password
- ✅ Onboarding fields in User model are now optional
- ✅ New `/api/auth/status` endpoint to check onboarding progress
- ✅ Onboarding endpoints now use session authentication (no userId in request body)

### What Stayed the Same
- ✅ Sign-in flow with NextAuth
- ✅ Dashboard and other features unchanged
- ✅ API response formats maintained

---

## Required Frontend Changes

### 1. Sign Up Flow - MINIMAL CHANGES

**Current Sign Up Page**: `src/app/auth/signup/page.tsx`

**Change**: Remove any default/placeholder values for onboarding fields

**Before**:
```typescript
// If you were sending onboarding data during signup
const response = await fetch('/api/auth/signup', {
  method: 'POST',
  body: JSON.stringify({
    email,
    password,
    fullName: '',  // ❌ Don't send this anymore
    location: {},  // ❌ Don't send this anymore
    // ... other onboarding fields
  }),
});
```

**After**:
```typescript
// Only send login credentials
const response = await fetch('/api/auth/signup', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    email,
    password,
  }),
});

if (response.ok) {
  const { data } = await response.json();
  // Automatically sign in the user
  await signIn('credentials', {
    email,
    password,
    callbackUrl: '/onboarding/initial-info', // Redirect to onboarding
  });
}
```

---

### 2. Sign In Flow - ADD ONBOARDING CHECK

**Current Sign In Page**: `src/app/auth/signin/page.tsx`

**Change**: After successful login, check if onboarding is complete

**Add this logic after sign-in**:
```typescript
const handleSignIn = async (email: string, password: string) => {
  const result = await signIn('credentials', {
    email,
    password,
    redirect: false, // Handle redirect manually
  });

  if (result?.ok) {
    // Check onboarding status
    const statusRes = await fetch('/api/auth/status');
    const { data } = await statusRes.json();

    if (!data.hasCompletedOnboarding) {
      if (!data.hasProfileInfo) {
        router.push('/onboarding/initial-info');
      } else if (!data.hasStockSelection) {
        router.push('/onboarding/select-stocks');
      }
    } else {
      router.push(`/dashboard/${data.userId}`);
    }
  }
};
```

---

### 3. Onboarding Step 1 - REMOVE userId from Request

**Current Page**: `src/app/onboarding/initial-info/page.tsx`

**Change**: Don't send userId in the request body (backend uses session)

**Before**:
```typescript
const response = await fetch('/api/onboarding/initial-info', {
  method: 'POST',
  body: JSON.stringify({
    userId: session.user.id, // ❌ Remove this
    fullName,
    location,
    initialInvestmentAmount,
    savingsThreshold,
    annualSavingsInterestRate,
  }),
});
```

**After**:
```typescript
const response = await fetch('/api/onboarding/initial-info', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    // userId removed - backend gets it from session
    fullName,
    location,
    initialInvestmentAmount,
    savingsThreshold,
    annualSavingsInterestRate,
  }),
});

if (response.ok) {
  const { data } = await response.json();
  // Move to stock selection
  router.push('/onboarding/select-stocks');
}
```

---

### 4. Onboarding Step 2 - REMOVE userId from Request

**Current Page**: `src/app/onboarding/select-stocks/page.tsx`

**Change**: Don't send userId in the request body

**Before**:
```typescript
const response = await fetch('/api/onboarding/select-stocks', {
  method: 'POST',
  body: JSON.stringify({
    userId: session.user.id, // ❌ Remove this
    selectedStockIds,
  }),
});
```

**After**:
```typescript
const response = await fetch('/api/onboarding/select-stocks', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    // userId removed - backend gets it from session
    selectedStockIds,
  }),
});

if (response.ok) {
  const { data } = await response.json();
  // Onboarding complete! Redirect to dashboard
  router.push(`/dashboard/${data.userId}`);
}
```

---

### 5. App-Level Auth Check - ADD STATUS CHECK (RECOMMENDED)

**Create or Update**: `src/components/AuthGuard.tsx` or app-level middleware

**Purpose**: Check auth status on app load and route users appropriately

```typescript
'use client';

import { useSession } from 'next-auth/react';
import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { data: session, status } = useSession();
  const router = useRouter();
  const pathname = usePathname();
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    async function checkOnboardingStatus() {
      if (status === 'loading') return;

      if (status === 'unauthenticated') {
        // Not logged in
        if (!pathname.startsWith('/auth')) {
          router.push('/auth/signin');
        }
        setIsChecking(false);
        return;
      }

      // User is authenticated, check onboarding status
      try {
        const response = await fetch('/api/auth/status');
        const { data } = await response.json();

        if (!data.hasCompletedOnboarding) {
          // Redirect to appropriate onboarding step
          if (!data.hasProfileInfo && pathname !== '/onboarding/initial-info') {
            router.push('/onboarding/initial-info');
          } else if (!data.hasStockSelection && pathname !== '/onboarding/select-stocks') {
            router.push('/onboarding/select-stocks');
          }
        } else if (pathname.startsWith('/onboarding')) {
          // Already completed onboarding, redirect to dashboard
          router.push(`/dashboard/${data.userId}`);
        }
      } catch (error) {
        console.error('Failed to check auth status:', error);
      } finally {
        setIsChecking(false);
      }
    }

    checkOnboardingStatus();
  }, [status, pathname, router]);

  if (isChecking) {
    return <div>Loading...</div>; // Or your loading component
  }

  return <>{children}</>;
}
```

**Usage in Layout**:
```typescript
// src/app/layout.tsx
import { AuthGuard } from '@/components/AuthGuard';

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        <SessionProvider>
          <AuthGuard>
            {children}
          </AuthGuard>
        </SessionProvider>
      </body>
    </html>
  );
}
```

---

## Optional: Use the New Status Endpoint

### Get User Status Anywhere

```typescript
async function getUserStatus() {
  const response = await fetch('/api/auth/status');
  const { data } = await response.json();
  
  return {
    isAuthenticated: data.isAuthenticated,
    hasCompletedOnboarding: data.hasCompletedOnboarding,
    hasProfileInfo: data.hasProfileInfo,
    hasStockSelection: data.hasStockSelection,
    userId: data.userId,
    email: data.email,
    fullName: data.fullName,
  };
}
```

### Show Progress in UI

```typescript
function OnboardingProgress() {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    async function loadStatus() {
      const data = await getUserStatus();
      setStatus(data);
    }
    loadStatus();
  }, []);

  if (!status) return null;

  return (
    <div className="flex gap-2">
      <Step completed={status.hasProfileInfo} label="Profile Info" />
      <Step completed={status.hasStockSelection} label="Stock Selection" />
    </div>
  );
}
```

---

## Summary of Changes by File

| File | Change Required | Complexity |
|------|----------------|------------|
| `auth/signup/page.tsx` | Remove onboarding fields from request | Low |
| `auth/signin/page.tsx` | Add status check after login | Low |
| `onboarding/initial-info/page.tsx` | Remove userId from request | Very Low |
| `onboarding/select-stocks/page.tsx` | Remove userId from request | Very Low |
| `components/AuthGuard.tsx` (new) | Add status-based routing | Medium (Optional) |

---

## Testing Checklist

### New User Flow
- [ ] Sign up with email/password
- [ ] Automatically logged in after signup
- [ ] Redirected to `/onboarding/initial-info`
- [ ] Submit profile information
- [ ] Redirected to `/onboarding/select-stocks`
- [ ] Select stocks
- [ ] Redirected to dashboard
- [ ] Refresh page - should stay on dashboard (onboarding complete)

### Existing User Flow (if applicable)
- [ ] Sign in with existing credentials
- [ ] If onboarding incomplete, redirected to appropriate step
- [ ] If onboarding complete, redirected to dashboard

### Error Cases
- [ ] Sign up with existing email - shows error
- [ ] Sign in with wrong password - shows error
- [ ] Try to access dashboard without completing onboarding - redirected
- [ ] Try to submit stock selection before profile info - shows error

---

## Rollback Plan

If you need to rollback to the old flow:

1. Revert User model changes (make fields required again)
2. Update signup route to include default onboarding values
3. Remove `/api/auth/status` endpoint
4. Add userId back to onboarding request bodies

All changes are backwards compatible - existing code will continue to work, just needs the small adjustments listed above.

---

## Support

### Common Issues

**Issue**: "Unauthorized" error on onboarding endpoints
- **Fix**: Ensure user is logged in and session is valid. Check NextAuth configuration.

**Issue**: User stuck in onboarding loop
- **Fix**: Check `/api/auth/status` response. Ensure all required fields are being saved.

**Issue**: Can't access dashboard after completing onboarding
- **Fix**: Verify `portfolio.onboardingComplete` is set to `true` in database.

### Need Help?

Check the full API documentation in `API_AUTH_ONBOARDING.md` for:
- Detailed endpoint specifications
- Request/response examples
- Error codes and messages
- Security considerations

---

## Benefits of New Flow

✅ **Better UX**: Users can create account and login quickly, complete profile later  
✅ **Cleaner separation**: Login concerns separate from business data  
✅ **More secure**: Session-based authentication, userId can't be spoofed  
✅ **Flexible**: Easy to add more onboarding steps in the future  
✅ **Type-safe**: Optional fields properly typed in TypeScript  
✅ **Better error handling**: Clear validation at each step  

---

## Next Steps

1. Update the 4 main files listed above (signup, signin, onboarding steps)
2. Test the complete flow with a new user account
3. (Optional) Add the AuthGuard component for better UX
4. (Optional) Add progress indicators in onboarding UI
5. Deploy and monitor for any issues

The changes are minimal and focused - your existing dashboard and other features don't need any modifications!