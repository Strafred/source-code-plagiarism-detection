public class Kasus4 {
    public static double Konversi(int miles) {
            return miles * 1.609;
    }
    public static void main(String[] args) {
        System.out.println("Miles\t\tKilometers");
        System.out.println("-------------------------------");
        for(int miles=1 ; miles<=10 ; miles++)
            System.out.println(miles + "\t\t" + Konversi(miles));
    }
}
