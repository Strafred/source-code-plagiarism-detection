public class Kasus4 {
    public static double Konversi(int miles) {
        if(miles <=10) {
            System.out.println(miles + "\t\t" + miles * 1.609);
            return Konversi(++miles);
        }
        return 0;
    }
    public static void main(String[] args) {
        System.out.println("Miles\t\tKilometers");
        System.out.println("-------------------------------");
        Konversi(1);
    }
}
