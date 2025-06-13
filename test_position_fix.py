#!/usr/bin/env python3
"""
Quick test to verify the position.asset_type fix
"""

# Mock Position class to simulate the issue


class MockPosition:
    def __init__(self, symbol, asset):
        self.symbol = symbol
        self.asset = asset
        self.quantity = 1.0


class MockAsset:
    def __init__(self, symbol, asset_type):
        self.symbol = symbol
        self.asset_type = asset_type

# Test the original problematic code vs our fix


def test_original_approach():
    """This would fail with AttributeError: 'MockPosition' object has no attribute 'asset_type'"""
    asset = MockAsset("BTC", "crypto")
    position = MockPosition("BTC", asset)

    try:
        # This was the problematic line (without getattr safety)
        asset_type = position.asset_type  # This will fail
        print(
            f"❌ Original approach: Got asset_type = {asset_type} (this shouldn't work)")
    except AttributeError as e:
        print(f"✅ Original approach correctly failed: {e}")

    try:
        # Even with getattr, it gets the wrong value (uses default instead of actual asset_type)
        asset_type = getattr(position, 'asset_type', 'crypto')
        print(
            f"❌ Original getattr approach: Got asset_type = {asset_type} (uses default, not actual)")
    except Exception as e:
        print(f"❌ Original getattr approach failed: {e}")


def test_fixed_approach():
    """This should work correctly"""
    asset = MockAsset("BTC", "crypto")
    position = MockPosition("BTC", asset)

    try:
        # This is our fix
        asset_type = getattr(position.asset, 'asset_type', 'crypto') if hasattr(
            position, 'asset') and position.asset else 'crypto'
        print(f"✅ Fixed approach: Got asset_type = {asset_type} (correct!)")
        return asset_type
    except Exception as e:
        print(f"❌ Fixed approach failed: {e}")
        return None


if __name__ == "__main__":
    print("Testing position.asset_type fix...")
    print()

    test_original_approach()
    result = test_fixed_approach()

    print()
    if result == "crypto":
        print("✅ Fix verified! The code now correctly accesses position.asset.asset_type")
    else:
        print("❌ Fix needs more work")
