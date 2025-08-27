import Metal
import Foundation

/// キャッシュキーのプロトコル
internal protocol CacheKey: Hashable, CustomStringConvertible {}

/// テクスチャキャッシュを管理するクラス
/// LRU (Least Recently Used) アルゴリズムを使用してメモリ効率的にキャッシュを管理
internal class TextureCacheManager<T> {
    
    // MARK: - Properties
    
    private let queue: DispatchQueue
    private var cache: [String: T] = [:]
    private var accessOrder: [String] = []
    private let maxCacheSize: Int
    private let cacheName: String
    
    // MARK: - Initialization
    
    /// テクスチャキャッシュマネージャーを初期化
    /// - Parameters:
    ///   - maxSize: 最大キャッシュサイズ（デフォルト: 3）
    ///   - name: キャッシュの名前（ログ用）
    internal init(maxSize: Int = 2, name: String = "TextureCache") {
        self.maxCacheSize = maxSize
        self.cacheName = name
        self.queue = DispatchQueue(
            label: "com.metalfluid.texturecache.\(name.lowercased())",
            attributes: .concurrent
        )
    }
    
    deinit {
        let cacheName = self.cacheName
        let cacheCount = self.cache.count
        queue.async(flags: .barrier) {
            print("🗑️ \(cacheName): Cache manager deallocated, had \(cacheCount) items")
        }
    }
    
    // MARK: - Cache Operations
    
    /// キャッシュからアイテムを取得、または作成関数を実行してキャッシュに追加
    /// - Parameters:
    ///   - key: キャッシュキー
    ///   - createItem: アイテムが見つからない場合の作成関数
    /// - Returns: キャッシュされたアイテムまたは新しく作成されたアイテム
    internal func getOrCreate<K: CacheKey>(
        key: K,
        createItem: () throws -> T
    ) rethrows -> T {
        let keyString = key.description
        
        // 読み取り操作（concurrent）
        if let cachedItem = queue.sync(execute: {
            return cache[keyString]
        }) {
            // アクセス順序を更新（barrier write）
            queue.async(flags: .barrier) {
                self.updateAccessOrder(keyString)
            }
            print("💾 \(cacheName): Cache hit for key '\(keyString)'")
            return cachedItem
        }
        
        // キャッシュミス - アイテムを作成
        print("🔄 \(cacheName): Cache miss for key '\(keyString)', creating new item")
        let newItem = try createItem()
        
        // キャッシュに追加（barrier write）
        queue.async(flags: .barrier) {
            self.addToCache(keyString, item: newItem)
        }
        
        return newItem
    }
    
    /// キャッシュをクリア
    internal func clearCache() {
        queue.async(flags: .barrier) {
            let clearedCount = self.cache.count
            self.cache.removeAll()
            self.accessOrder.removeAll()
            if clearedCount > 0 {
                print("🗑️ \(self.cacheName): Manually cleared \(clearedCount) cached items")
            }
        }
    }
    
    /// メモリ警告時の処理（最新のエントリのみ保持）
    internal func handleMemoryWarning() {
        queue.async(flags: .barrier) {
            let originalCount = self.cache.count
            
            // 最新のキャッシュエントリのみを保持
            if self.accessOrder.count > 1 {
                let mostRecentKey = self.accessOrder.last!
                let mostRecentItem = self.cache[mostRecentKey]
                
                self.cache.removeAll()
                self.accessOrder.removeAll()
                
                if let item = mostRecentItem {
                    self.cache[mostRecentKey] = item
                    self.accessOrder.append(mostRecentKey)
                }
                
                let clearedCount = originalCount - self.cache.count
                if clearedCount > 0 {
                    print("⚠️ \(self.cacheName): Memory warning - Cleared \(clearedCount) items, keeping most recent")
                }
            }
        }
    }
    
    /// キャッシュの情報を取得
    /// - Returns: キャッシュ数とキーのリスト
    internal func getCacheInfo() -> (count: Int, keys: [String]) {
        return queue.sync {
            return (cache.count, Array(cache.keys))
        }
    }
    
    // MARK: - Private Methods
    
    private func addToCache(_ key: String, item: T) {
        // 既存のエントリを削除（もしあれば）
        if let existingIndex = accessOrder.firstIndex(of: key) {
            accessOrder.remove(at: existingIndex)
        }
        
        // 新しいエントリを追加
        cache[key] = item
        accessOrder.append(key)
        
        // キャッシュサイズの管理
        manageCacheSize()
        
        print("💾 \(cacheName): Added item to cache (key: '\(key)', total: \(cache.count)/\(maxCacheSize))")
    }
    
    private func updateAccessOrder(_ key: String) {
        if let index = accessOrder.firstIndex(of: key) {
            accessOrder.remove(at: index)
            accessOrder.append(key)
        }
    }
    
    private func manageCacheSize() {
        while cache.count > maxCacheSize && !accessOrder.isEmpty {
            let oldestKey = accessOrder.removeFirst()
            cache.removeValue(forKey: oldestKey)
            print("🗑️ \(cacheName): LRU evicted item with key '\(oldestKey)' (cache: \(cache.count)/\(maxCacheSize))")
        }
    }
}

// MARK: - Screen Size Cache Key

/// 画面サイズをキャッシュキーとして使用するための構造体
internal struct ScreenSizeCacheKey: CacheKey {
    internal let size: SIMD2<Float>
    
    internal init(_ size: SIMD2<Float>) {
        self.size = size
    }
    
    internal var description: String {
        return "\(Int(size.x))x\(Int(size.y))"
    }
    
    internal func hash(into hasher: inout Hasher) {
        hasher.combine(Int(size.x))
        hasher.combine(Int(size.y))
    }
    
    internal static func == (lhs: ScreenSizeCacheKey, rhs: ScreenSizeCacheKey) -> Bool {
        return lhs.size.x == rhs.size.x && lhs.size.y == rhs.size.y
    }
}
