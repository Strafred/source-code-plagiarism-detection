public class Kasus4 {
    public static void Konversi(int n) {
        int miles = 1;
        while (miles <= n) {
            System.out.println(miles + "\t\t" + miles * 1.609);
            miles++;
        }
    }
    public static void main(String[] args) {
        System.out.println("Miles\t\tKilometers");
        System.out.println("-------------------------------");
        Konversi(10);
    }
}
